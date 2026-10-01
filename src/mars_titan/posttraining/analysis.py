"""Resumir las evaluaciones confirmadas por ventana, método y semilla."""

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json
from mars_titan.training.predictive_parents import _verified_file

from .inputs import fingerprint


def aggregate(rows):
    """Dar el mismo peso a cada ventana, sin tratar semillas o ventanas como independientes."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (row["stage"], row["family"], row["method"], row["partition"])
        grouped[key][row["fold"]].append(row)
    result = []
    for (stage, family, method, partition), folds in sorted(grouped.items()):
        means = [fmean(row["session_mae"] for row in values) for values in folds.values()]
        deltas = [
            fmean(row["delta_parent"] for row in values if row["delta_parent"] is not None)
            for values in folds.values()
            if any(row["delta_parent"] is not None for row in values)
        ]
        result.append(
            dict(
                stage=stage,
                family=family,
                method=method,
                partition=partition,
                folds=len(folds),
                runs=sum(len(values) for values in folds.values()),
                mean_fold_session_mae=fmean(means),
                minimum_fold_session_mae=min(means),
                maximum_fold_session_mae=max(means),
                mean_fold_delta_parent=fmean(deltas) if deltas else None,
            )
        )
    return result


def analyse_completion(output):
    """Publicar un informe local solo tras confirmar todas las etapas y sus recibos."""
    output = Path(output)
    summary, _ = read_manifest(output / "summary.json", 8 * 1024**2)
    stages = summary["stages"]
    if (
        summary.get("final_test_opened") is not False
        or not stages
        or any(
            row["status"] != "completed" or row["completed_runs"] != row["planned_runs"]
            for row in stages
        )
    ):
        raise ValueError("El análisis necesita todas las etapas completas")
    rows, sources = [], []
    for stage in stages:
        path = output / stage["fold"] / stage["stage"] / "summary.json"
        report, signature = read_manifest(path, 8 * 1024**2)
        if (
            signature != stage["sha256"]
            or report["status"] != "completed"
            or report["final_test_opened"] is not False
        ):
            raise ValueError("Una etapa confirmada ha cambiado")
        sources.append(dict(path=str(path.relative_to(output)), sha256=signature))
        if stage["stage"] != "evaluation":
            continue
        for identifier, record in report["runs"].items():
            receipt, _ = _verified_file(path.parent, record, maximum_bytes=8 * 1024**2)
            run, _ = read_manifest(receipt, 8 * 1024**2)
            if run["status"] != "completed" or run["final_test_opened"] is not False:
                raise ValueError("La evaluación no conserva su reserva y estado")
            case = run.get("case") or {}
            method = case.get("mode", case.get("loss", "reference"))
            for partition, predictions in run["predictions"].items():
                if partition not in {"calibration", "evaluation"}:
                    raise ValueError("El análisis solo admite los bloques posteriores declarados")
                _verified_file(receipt.parent, predictions)
                metrics = predictions["metrics"]
                score = metrics["prediction"]["session_mae"]
                rows.append(
                    dict(
                        fold=stage["fold"],
                        id=identifier,
                        stage=run["job"]["stage"] + "/" + run["job"]["phase"],
                        family=run["family"],
                        method=method,
                        seed=case.get("seed"),
                        partition=partition,
                        samples=metrics["prediction"]["samples"],
                        session_mae=score,
                        session_mse=metrics["prediction"]["session_mse"],
                        delta_parent=(
                            score - metrics["parent"]["session_mae"]
                            if run["parent_comparison"]
                            else None
                        ),
                        delta_zero=score - metrics["zero"]["session_mae"],
                        primary=run["primary"],
                        best_epoch=(run.get("selection") or {}).get("best_epoch"),
                    )
                )
    if not rows:
        raise ValueError("Faltan resultados de evaluación")
    report = dict(
        schema_version=1,
        status="completed",
        created_at_utc=datetime.now(UTC).isoformat(),
        completion_identity_sha256=fingerprint(summary["identity"]),
        sources=sources,
        results=rows,
        aggregate=aggregate(rows),
        final_test_opened=False,
        selection_with_evaluation=False,
        limitations=[
            "Las medias son descriptivas y dan el mismo peso a cada ventana.",
            "Las semillas y ventanas con periodos compartidos no son observaciones independientes.",
            "Las medias de la búsqueda no equivalen al rendimiento de un modelo seleccionado.",
            "Las políticas publican la mediana discreta y las referencias su predicción continua.",
            "No se han estimado rentabilidad ni intervalos de confianza.",
            "El candidato MARS-TITAN no está implementado ni evaluado.",
        ],
    )
    atomic_json(output / "analysis.json", report)
    lines = [
        "# Evaluación de las ventanas temporales",
        "",
        f"Resultados confirmados: {len(rows)} pares de modelo y partición.",
        "",
        "MAE por sesión. Un delta negativo indica menor error que el padre correspondiente.",
        "Las medias agrupan los casos de cada ventana y después dan el mismo peso a cada ventana.",
        "",
        "| Etapa | Familia | Método | Partición | Ventanas | Casos | MAE medio | Delta al padre |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate"]:
        delta = row["mean_fold_delta_parent"]
        delta_text = "No procede" if delta is None else f"{delta:.8f}"
        lines.append(
            f"| {row['stage']} | {row['family']} | {row['method']} | {row['partition']} | "
            f"{row['folds']} | {row['runs']} | {row['mean_fold_session_mae']:.8f} | {delta_text} |"
        )
    lines.extend(
        [
            "",
            *report["limitations"],
            "",
            "El test final permanece cerrado. analysis.json conserva cada caso y su semilla,",
            "las métricas y las huellas de los recibos.",
            "",
        ]
    )
    temporary = output / ".analysis.md.tmp"
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(output / "analysis.md")
    return report
