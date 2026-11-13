"""Ventana de Ridge de la campaña de comparación (#836).

Ridge es la referencia lineal de la memoria. Se ajusta con el ejecutor de la campaña anterior
(`tabular_corpus.run_tabular_reference`), que recorre todas las filas de ajuste de la vista sin
reducir la población, estandariza cada entrada con los momentos del ajuste y resuelve en
float64. Esta ventana lo llama una vez por cada alpha declarado, elige el alpha con la
validación y deja un informe con la forma de las demás ventanas de la campaña.

Los alphas comparten las estadísticas del ajuste dentro del proceso, así que la vista se
recorre una sola vez para todos. Un ajuste cortado no deja nada aprovechable: el siguiente
intento escribe en otra carpeta y el anterior se conserva.

Al confirmar la ventana solo quedan las tablas de predicción del alpha elegido. De los demás
se conserva su informe con las métricas de cada partición, que basta para estudiar la
sensibilidad al alpha, y sus tablas se retiran para no ocupar disco.
"""

import math
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256

from .learning_hold import require_learning_allowed
from .reference_run import HELDOUT_FULL_TRAIN_SESSIONS
from .tabular_corpus import run_tabular_reference

KIND = "comparison_ridge_window_v1"
IDENTITY = "ridge_full_history_v1"
RULE = "menor MAE por sesión de validación y, con empate, el alpha mayor"
_REPORT_BYTES = 16 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _now():
    return datetime.now(UTC).isoformat()


def checked_config(config):
    """Comprobar la configuración declarada de Ridge y devolverla."""
    _require(
        isinstance(config, dict)
        and set(config) == {"identity", "alphas", "batch_size"}
        and config["identity"] == IDENTITY
        and isinstance(config["alphas"], list)
        and 1 <= len(config["alphas"]) <= 8
        and all(
            type(alpha) in (int, float) and math.isfinite(alpha) and alpha > 0
            for alpha in config["alphas"]
        )
        and len(set(config["alphas"])) == len(config["alphas"])
        and type(config["batch_size"]) is int
        and 1 <= config["batch_size"] <= 4096,
        "Ridge declara su identidad, alphas positivos y distintos y el tamaño de la tanda",
    )
    return config


def _fit(view, output, name, alpha, batch_size):
    """Ajustar un alpha en una carpeta de intento nueva y devolver su registro."""
    attempts = sorted((output / name).glob("attempt-*")) if (output / name).is_dir() else []
    folder = output / name / f"attempt-{len(attempts) + 1:04d}"
    folder.parent.mkdir(parents=True, exist_ok=True)
    fitted = run_tabular_reference(
        view,
        folder,
        kind="ridge",
        alpha=alpha,
        batch_size=batch_size,
        input_policy=HISTORICAL_MASKED,
        prediction_retention=HELDOUT_FULL_TRAIN_SESSIONS,
    )
    _require(fitted["status"] == "completed", f"Ridge no ha terminado el ajuste de {name}")
    return dict(
        alpha=alpha,
        path=folder.relative_to(output).as_posix(),
        run_sha256=sha256(folder / "run.json"),
        validation_session_mae=fitted["predictions"]["validation"]["metrics"]["session_mae"],
    )


def run_ridge_window(view, *, output, config, stop=None):
    """Ajustar Ridge con cada alpha, elegir el de menor MAE de validación y dejar su informe.

    La elección solo mira la validación. La calibración y la evaluación del alpha elegido se
    escribieron al ajustarlo y no intervienen. La misma llamada continúa una ventana pausada
    sin repetir los alphas ya confirmados.
    """
    require_learning_allowed("la ventana Ridge de la campaña de comparación")
    config = checked_config(config)
    view, output = Path(view), Path(output)
    safe_destination(output)
    request = dict(view_sha256=sha256(view), config=config)
    path = output / "run.json"
    if path.is_file():
        report = read_manifest(path, _REPORT_BYTES)[0]
        _require(report.get("request") == request, "La petición de la ventana ha cambiado")
        if report["status"] == "completed":
            return report
    else:
        report = dict(
            schema_version=1,
            kind=KIND,
            request=request,
            status="running",
            started_at_utc=_now(),
            final_test_opened=False,
            fits={},
        )
    output.mkdir(parents=True, exist_ok=True)
    for alpha in map(float, config["alphas"]):
        name = f"alpha-{alpha!r}"
        if name in report["fits"]:
            continue
        if stop is not None and stop.requested:
            report["status"] = "paused"
            atomic_json(path, report)
            return report
        report["status"] = "running"
        report["fits"][name] = _fit(view, output, name, alpha, config["batch_size"])
        atomic_json(path, report)
    chosen = min(
        report["fits"].values(), key=lambda fit: (fit["validation_session_mae"], -fit["alpha"])
    )
    fitted, digest = read_manifest(output / chosen["path"] / "run.json", _REPORT_BYTES)
    _require(digest == chosen["run_sha256"], "Ha cambiado el informe del alpha elegido")
    released = []
    for fit in report["fits"].values():
        if fit is not chosen:
            for table in sorted((output / fit["path"]).glob("*-predictions.parquet")):
                table.unlink()
                released.append(table.relative_to(output).as_posix())
    report.update(
        status="completed",
        finished_at_utc=_now(),
        selection=dict(alpha=chosen["alpha"], rule=RULE),
        released_tables=released,
        predictions={
            partition: dict(
                path=f"{chosen['path']}/{record['path']}",
                sha256=record["sha256"],
                metrics=record["metrics"],
            )
            for partition, record in fitted["predictions"].items()
        },
    )
    atomic_json(path, report)
    return report
