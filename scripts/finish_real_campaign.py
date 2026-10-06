"""Cerrar el análisis de una campaña real confirmada, con intentos acotados."""

import argparse
import fcntl
import os
import re
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.run_receipts import initialize_receipt

PARAMETERS = dict(repetitions=2000, seed=42, initial_cache_bytes=8 * 1024**2)
STAGES = (("neural", "run"), ("completion", "run"), ("reliability", "model"))
STATUSES = {"pending", "running", "paused", "blocked", "failed", "completed"}
COMPARISON_FILES = {
    "cases.csv",
    "methods.csv",
    "folds.csv",
    "intervals.csv",
    "session-errors.parquet",
}
EXPORT_FILES = {
    "predictive-cases.csv",
    "predictive-methods.csv",
    "predictive-folds.csv",
    "predictive-intervals.csv",
    "reliability-cases.csv",
    "predictive-periods.svg",
    "predictive-periods.png",
}
MAX_ARTIFACT_BYTES = 512 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _state(path):
    safe_destination(path)
    report, signature = read_manifest(path, 8 * 1024**2)
    _require(
        isinstance(report, dict) and isinstance(report.get("identity"), dict),
        "El resumen no conserva una identidad válida",
    )
    identity = report.get("identity", {})
    paths = identity.get("paths", {})
    _require(isinstance(paths, dict), "Las rutas del coordinador no son válidas")
    stages = report.get("stages", [])
    _require(
        report.get("schema_version") == 1
        and report.get("kind") == "real_retraining_campaign"
        and report.get("domain") == "real"
        and report.get("status") in STATUSES
        and report.get("final_test_opened") is False
        and identity.get("final_test_opened") is False
        and isinstance(paths.get("state_dir"), str)
        and Path(paths["state_dir"]).is_absolute()
        and Path(paths["state_dir"]).resolve() == path.parent.resolve(),
        "El resumen no acredita una campaña real, su directorio y la reserva cerrada",
    )
    _require(isinstance(stages, list) and len(stages) == 3, "Faltan las tres etapas previstas")
    for row, (name, unit) in zip(stages, STAGES, strict=True):
        _require(
            isinstance(row, dict)
            and row.get("name") == name
            and row.get("unit") == unit
            and row.get("status") in STATUSES
            and type(row.get("planned")) is int
            and type(row.get("completed")) is int
            and 0 <= row["completed"] <= row["planned"]
            and row["planned"] > 0
            and (row["status"] != "completed" or row["completed"] == row["planned"]),
            "Una etapa no conserva su identidad, estado o recuentos",
        )
    counts = {
        "planned_runs": sum(row["planned"] for row in stages[:2]),
        "completed_runs": sum(row["completed"] for row in stages[:2]),
        "planned_reliability_models": stages[2]["planned"],
        "completed_reliability_models": stages[2]["completed"],
    }
    _require(
        all(type(report.get(key)) is int and report[key] == value for key, value in counts.items()),
        "Los recuentos del coordinador no concilian",
    )
    _require(
        report["status"] != "completed" or all(row["status"] == "completed" for row in stages),
        "El cierre conserva etapas pendientes",
    )
    return report, signature


def _artifact(path, digest):
    safe_destination(path)
    _require(
        isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest) is not None,
        "Falta una huella SHA-256 válida",
    )
    _require(
        path.is_file() and path.stat().st_size <= MAX_ARTIFACT_BYTES,
        "Falta un artefacto regular o supera su límite",
    )
    _require(sha256(path) == digest, "Un artefacto ha cambiado desde su confirmación")
    return digest


def _inputs(state, report, output):
    paths = report["identity"]["paths"]
    _require(
        all(
            isinstance(paths.get(name), str) and Path(paths[name]).is_absolute()
            for name in ("references", "completion", "state_dir")
        ),
        "Faltan las rutas absolutas de la campaña",
    )
    for name, value in paths.items():
        _require(
            isinstance(value, str) and Path(value).is_absolute(),
            "Una ruta de entrada no es absoluta",
        )
        source = Path(value)
        safe_destination(source)
        if name in {"encoded", "neural_config", "tabular_config", "post_config"}:
            source = source.parent
        outside_source(source, output)
        outside_source(output, source)
    sources = dict(
        reference=Path(paths["references"]),
        completion=Path(paths["completion"]),
        reliability=state.parent / "reliability",
    )
    files = [
        sources["reference"] / "summary.json",
        sources["completion"] / "summary.json",
        sources["reliability"] / "reliability.json",
    ]
    kinds = (
        "temporal_reference_search",
        "temporal_posttraining_completion",
        "frozen_campaign_reliability",
    )
    hashes, receipts = {}, []
    for index, (path, progress, kind) in enumerate(
        zip(files, report["stages"], kinds, strict=True)
    ):
        safe_destination(path)
        receipt, signature = read_manifest(path, (64 if index == 2 else 8) * 1024**2)
        _require(
            signature == progress.get("sha256")
            and receipt.get("kind") == kind
            and receipt.get("status") == "completed"
            and receipt.get("final_test_opened") is False,
            "Una etapa no conserva su cierre o huella",
        )
        if index < 2:
            _require(
                type(receipt.get("planned_runs")) is int
                and type(receipt.get("completed_runs")) is int
                and receipt["planned_runs"] == receipt["completed_runs"] == progress["planned"],
                "Los recuentos de una etapa no coinciden con el coordinador",
            )
        receipts.append(receipt)
        hashes[progress["name"]] = signature
    reliable = receipts[2]
    provenance = reliable.get("provenance", {})
    models = report["planned_reliability_models"]
    _require(
        receipts[1].get("identity", {}).get("reference_sha256") == hashes["neural"]
        and provenance.get("reference_sha256") == hashes["neural"]
        and provenance.get("completion_sha256") == hashes["completion"]
        and provenance.get("domain") == "real"
        and provenance.get("final_test_opened") is False
        and reliable.get("target_kind") == "residual_return"
        and reliable.get("coverage_guaranteed") is False
        and reliable.get("counts", {}).get("models") == models
        and reliable["counts"].get("prediction_files") == 2 * models
        and models <= 4096,
        "La fiabilidad y continuación no pertenecen a los mismos modelos y fuentes",
    )
    _require(
        set(reliable.get("artifacts", {})) == {"cases.csv"},
        "La fiabilidad tiene artefactos incompatibles",
    )
    hashes["reliability_cases"] = _artifact(
        sources["reliability"] / "cases.csv", reliable["artifacts"]["cases.csv"]
    )
    return sources, hashes


def _code():
    import mars_titan

    root = Path(mars_titan.__file__).parent
    files = sorted(root.rglob("*.py"))
    _require(0 < len(files) <= 1024, "El código excede el presupuesto de identidad")
    return dict(
        package={path.relative_to(root).as_posix(): sha256(path) for path in files},
        scripts={
            name: sha256(Path(__file__).with_name(name))
            for name in ("finish_real_campaign.py", "export_campaign_comparison.py")
        },
    )


def _identity(state, report, output):
    sources, hashes = _inputs(state, report, output)
    identity = dict(
        coordinator={key: value for key, value in report.items() if key != "updated_at_utc"},
        paths=dict(state=str(state.resolve()), output=str(output.resolve())),
        sources=hashes,
        code=_code(),
        parameters=dict(PARAMETERS),
        versions={name: version(name) for name in ("numpy", "scipy", "pyarrow", "matplotlib")},
    )
    return identity, sources


def _verified_report(directory, name, expected_files):
    safe_destination(directory / name)
    report, signature = read_manifest(directory / name, 64 * 1024**2)
    artifacts = report.get("artifacts", {})
    _require(
        isinstance(artifacts, dict) and set(artifacts) == expected_files,
        "El resultado no conserva todos sus artefactos",
    )
    hashes = {name: signature}
    for filename, digest in artifacts.items():
        hashes[filename] = _artifact(directory / filename, digest)
    return report, hashes


def _result(directory, identity):
    comparison, first = _verified_report(
        directory / "comparison", "comparison.json", COMPARISON_FILES
    )
    evidence, second = _verified_report(directory / "export", "evidence.json", EXPORT_FILES)
    sources = identity["sources"]
    provenance = comparison.get("provenance", {})
    code = comparison.get("analysis_source_sha256")
    _require(
        isinstance(code, dict)
        and set(code)
        == {
            "evaluation/comparison_sources.py",
            "evaluation/prediction_statistics.py",
            "evaluation/campaign_comparison.py",
        }
        and all(
            name in identity["code"]["package"] and identity["code"]["package"][name] == digest
            for name, digest in code.items()
        ),
        "La comparación no conserva la identidad de su código",
    )
    _require(
        evidence.get("renderer", {}).get("script_sha256")
        == identity["code"]["scripts"]["export_campaign_comparison.py"]
        and evidence.get("predictive", {}).get("final_test_opened") is False
        and evidence.get("reliability", {}).get("final_test_opened") is False
        and evidence["predictive"].get("source_artifacts")
        == {name: first[name] for name in COMPARISON_FILES if name != "session-errors.parquet"}
        and evidence["reliability"].get("source_artifacts")
        == {"cases.csv": sources["reliability_cases"]},
        "La exportación no conserva el código, la reserva o los artefactos de sus productores",
    )
    models = identity["coordinator"]["planned_reliability_models"]
    _require(
        comparison.get("status") == "completed"
        and comparison.get("final_test_opened") is False
        and comparison.get("counts", {}).get("models") == models
        and comparison["counts"].get("prediction_files") == 2 * models
        and provenance.get("reference_sha256") == sources["neural"]
        and provenance.get("completion_sha256") == sources["completion"]
        and provenance.get("domain") == "real"
        and provenance.get("final_test_opened") is False
        and all(
            comparison.get("method", {}).get(key) == PARAMETERS[key]
            for key in ("seed", "repetitions")
        )
        and evidence.get("predictive", {}).get("report_sha256") == first["comparison.json"]
        and evidence.get("reliability", {}).get("report_sha256") == sources["reliability"],
        "La comparación exportada no corresponde a esta campaña y sus parámetros",
    )
    return dict(comparison=first, export=second)


def _ledger(report, identity):
    attempts = report.get("attempts", [])
    _require(
        report.get("schema_version") == 1
        and report.get("kind") == "real_campaign_final_analysis"
        and report.get("identity") == identity
        and isinstance(attempts, list)
        and len(attempts) <= 3
        and report.get("status") in {"running", "failed", "interrupted", "completed"},
        "El recibo de análisis no conserva su identidad o estado",
    )
    for number, attempt in enumerate(attempts, 1):
        _require(
            isinstance(attempt, dict)
            and type(attempt.get("number")) is int
            and attempt["number"] == number
            and attempt.get("status") in {"running", "failed", "interrupted", "completed"}
            and (number == len(attempts) or attempt["status"] in {"failed", "interrupted"}),
            "El historial de intentos no es coherente",
        )
    _require(
        report["status"] != "completed" or attempts and attempts[-1]["status"] == "completed",
        "Falta el intento completo del análisis",
    )
    _require(
        not attempts or report["status"] == attempts[-1]["status"],
        "El estado no coincide con su último intento",
    )


def _compare(reference, completion, output):
    from mars_titan.evaluation.campaign_comparison import compare_campaigns

    return compare_campaigns(reference, completion, output, **PARAMETERS)


def _export(comparison, reliability, output):
    if __package__:
        from .export_campaign_comparison import main as export
    else:
        from export_campaign_comparison import main as export

    return export(
        [
            "--predictive",
            str(comparison),
            "--reliability",
            str(reliability),
            "--output",
            str(output),
        ]
    )


def finish_campaign(state, output):
    state, output = Path(state), Path(output)
    report, signature = _state(state)
    if report["status"] != "completed":
        return dict(status="waiting", campaign_status=report["status"], state_sha256=signature)
    safe_destination(output)
    identity, sources = _identity(state, report, output)
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".analysis.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(
            output, identity, record="summary.json", lock=".analysis.lock"
        )
        summary = (
            read_manifest(output / "summary.json", 8 * 1024**2)[0]
            if confirmed
            else dict(
                schema_version=1,
                kind="real_campaign_final_analysis",
                identity=identity,
                status="running",
                attempts=[],
                observed_state_sha256=signature,
            )
        )
        _ledger(summary, identity)
        attempts = summary["attempts"]
        if summary["status"] == "completed":
            directory = output / f"attempt-{len(attempts):03d}"
            try:
                _require(
                    attempts[-1].get("artifacts") == _result(directory, identity),
                    "El resultado completo perdió sus huellas",
                )
                _require(
                    _identity(state, _state(state)[0], output)[0] == identity,
                    "Las fuentes o el código cambiaron durante la verificación",
                )
            except Exception as error:
                attempts[-1].update(
                    status="failed", error=dict(type=type(error).__name__, message=str(error))
                )
                summary["status"] = "failed"
                atomic_json(output / "summary.json", summary)
                raise
            return summary
        if attempts and attempts[-1]["status"] == "running":
            attempts[-1].update(
                status="interrupted",
                error=dict(
                    type="InterruptedRun",
                    message="El intento anterior terminó sin confirmar su resultado",
                ),
                interruption_observed_at_utc=datetime.now(UTC).isoformat(),
            )
            summary["status"] = "interrupted"
            atomic_json(output / "summary.json", summary)
        _require(len(attempts) < 3, "Se agotaron los tres intentos de análisis")
        number = len(attempts) + 1
        attempt = dict(
            number=number, status="running", started_at_utc=datetime.now(UTC).isoformat()
        )
        attempts.append(attempt)
        summary.update(status="running", observed_state_sha256=signature)
        atomic_json(output / "summary.json", summary)
        directory = output / f"attempt-{number:03d}"
        try:
            directory.mkdir()
            _compare(sources["reference"], sources["completion"], directory / "comparison")
            _export(directory / "comparison", sources["reliability"], directory / "export")
            artifacts = _result(directory, identity)
            _require(
                _identity(state, _state(state)[0], output)[0] == identity,
                "Las fuentes o el código cambiaron durante el análisis",
            )
            attempt.update(status="completed", artifacts=artifacts)
            summary["status"] = "completed"
        except BaseException as error:
            status = (
                "interrupted"
                if isinstance(error, (KeyboardInterrupt, InterruptedError))
                else "failed"
            )
            attempt.update(status=status, error=dict(type=type(error).__name__, message=str(error)))
            summary["status"] = status
            raise
        finally:
            attempt["finished_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(output / "summary.json", summary)
        return summary
    finally:
        os.close(descriptor)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = finish_campaign(args.state, args.output)
    print(f"Estado del análisis: {result['status']}.")
    return 3 if result["status"] == "waiting" else 0


if __name__ == "__main__":
    raise SystemExit(main())
