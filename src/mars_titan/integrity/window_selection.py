"""Cada estado elegido sale de la validación anterior a la ventana que evalúa.

La comparación solo es honesta si estados, búsquedas, tasas y adaptadores se eligen con una
validación temporal anterior a la evaluación, nunca con 2024 ni con la propia evaluación.
Esta comprobación lo repite por otro camino a partir de los recibos confirmados:

1. En cada ventana reentrenada, la búsqueda elegida de cada modelo es la de menor MAE de
   validación, con empate por identificador, y esa puntuación es la que publica el informe
   del ajuste. Cada finalista repite el caso elegido con su semilla.
2. Cada recibo de ventana de un modelo comparado nombra el estado elegido para su semilla
   y declara una última etiqueta usada anterior al inicio de la evaluación
   (`read_window_receipt`).
3. Los tramos de cada ventana van en orden (validación, calibración y evaluación) y la
   comparación se publica sobre la evaluación, con la calibración como único tramo previo.
4. Con el walk-forward por etapas, el verificador de disjunción de #437 recorre vistas,
   recibos base y selecciones de la cadena sin fallos: las filas nuevas no cortan las del
   padre, toda etiqueta del padre madura antes de la primera decisión nueva y ninguna fila
   de 2024 aparece en las vistas.

No lee predicciones por fila, así que sirve también después de la retención v2.
"""

import json
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest
from mars_titan.environments.walk_forward_receipt import read_window_receipt
from mars_titan.evaluation.splits import PARTITIONS
from mars_titan.training.campaign_plan import FIT, ONLINE, PLATEAU

KIND = "window_selection"


def _receipt(output, job_id):
    path = Path(output) / "jobs" / job_id / "receipt.json"
    return read_manifest(path, 8 * 1024**2)[0] if path.is_file() else None


def _validation_score(output, receipt):
    """MAE por sesión de la validación según el informe que confirma el recibo."""
    record = receipt["report"]
    report, _ = read_manifest(Path(output) / record["path"], 256 * 1024**2)
    return report["predictions"]["validation"]["metrics"]["session_mae"]


def _groups(jobs):
    """Trabajos que eligen estado por ámbito, ventana y modelo."""
    groups = {}
    for job in jobs:
        if job["kind"] not in (FIT, ONLINE) or job.get("phase") == PLATEAU:
            continue
        key = (job["scope"], job["window"], job["arm"])
        groups.setdefault(key, []).append(job)
    return groups


def _searches(output, members, failures):
    """Búsqueda elegida de un grupo y comprobación de que su puntuación es la del informe."""
    scored = []
    for job in members:
        if job["stage"] != "search":
            continue
        receipt = _receipt(output, job["id"])
        if receipt is None:
            failures.append(f"{job['id']}: búsqueda sin recibo")
            continue
        published = _validation_score(output, receipt)
        if receipt["score"] != published:
            failures.append(f"{job['id']}: la puntuación del recibo no es la de su validación")
        scored.append((receipt["score"], job["id"], receipt))
    if not scored:
        return None
    _, key, receipt = min(scored, key=lambda item: (item[0], item[1]))
    return dict(job=key, receipt=receipt, searches=len(scored))


def _selected(output, jobs, scope, window, arm, seed, winner):
    """Recibo del estado elegido para una semilla: traslado, finalista o búsqueda ganadora."""
    for stage in ("carry", "finalist"):
        found = [
            job
            for job in jobs
            if (job["scope"], job["window"], job["arm"], job["seed"], job["stage"])
            == (scope, window, arm, seed, stage)
            and job.get("phase") != PLATEAU
        ]
        if found:
            return found[0]["id"], _receipt(output, found[0]["id"])
    if winner is not None and winner["receipt"]["identity"]["seed"] == seed:
        return winner["job"], winner["receipt"]
    return None, None


def check_selection(campaign, jobs, output):
    """Búsquedas, finalistas y recibos de ventana de los modelos comparados."""
    failures, chosen = [], {}
    compared = campaign["comparison_config"]["arms"]
    groups = _groups(jobs)
    winners = {}
    for key, members in groups.items():
        winner = _searches(output, members, failures)
        if winner is None:
            continue
        winners[key] = winner
        case = winner["receipt"]["identity"]["case"]
        for job in members:
            if job["stage"] != "finalist":
                continue
            receipt = _receipt(output, job["id"])
            expected = case | dict(seed=job["seed"])
            if receipt is None or receipt["identity"]["case"] != expected:
                failures.append(f"{job['id']}: el finalista no repite el caso elegido")
        chosen["/".join(key)] = dict(job=winner["job"], searches=winner["searches"])
    receipts = 0
    seeds = sorted(
        {
            (job["scope"], job["window"], job["arm"], job["seed"])
            for job in jobs
            if job["arm"] in compared and job.get("phase") != PLATEAU
        }
    )
    for scope, window, arm, seed in seeds:
        label = f"{scope}/{window}/{arm}/seed-{seed}"
        anchor = next(
            job["anchor"]
            for job in jobs
            if (job["scope"], job["window"], job["arm"], job["seed"]) == (scope, window, arm, seed)
        )
        winner = winners.get((scope, anchor, arm))
        key, receipt = _selected(output, jobs, scope, window, arm, seed, winner)
        if receipt is None:
            failures.append(f"{label}: sin estado elegido confirmado")
            continue
        folder = Path(output) / "windows" / scope / window / arm / f"seed-{seed}"
        found = sorted(folder.glob("*.json")) if folder.is_dir() else []
        if not found:
            failures.append(f"{label}: sin recibos de ventana")
        for path in found:
            receipts += 1
            record = json.loads(path.read_text(encoding="utf-8"))
            try:
                read_window_receipt(record)
            except ValueError as error:
                failures.append(f"{label} {path.stem}: {error}")
                continue
            if record["parent"] != receipt["parent"]:
                failures.append(f"{label} {path.stem}: el recibo no nombra el estado elegido")
    return dict(
        searched_groups=len(chosen),
        window_receipts=receipts,
        chosen=chosen,
        failures=failures,
        passed=not failures,
    )


def _microseconds(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def check_partitions(config, scopes):
    """Orden de los tramos de cada ventana de `scopes` y tramos que usa la comparación."""
    failures = []
    if config["partition"] != "evaluation" or config["calibration"]["partition"] != "calibration":
        failures.append("La comparación no evalúa la evaluación con la calibración previa")
    windows = 0
    for scope in scopes:
        for window_id, window in config["resolved_scopes"][scope]["windows"].items():
            windows += 1
            bounds = [tuple(_microseconds(day) for day in window[name]) for name in PARTITIONS]
            ordered = all(
                earlier[1] <= later[0] for earlier, later in zip(bounds, bounds[1:], strict=False)
            )
            if not ordered:
                failures.append(f"{scope}/{window_id}: tramos fuera de orden")
    return dict(windows=windows, failures=failures, passed=not failures)


def check_chain(campaign, views, output, posttraining):
    """Disjunción por ventana del diseño por etapas (#437), sin ajustar nada."""
    from mars_titan.training.chain_disjunction import verify

    report = verify(campaign, views, campaign_output=output, posttraining=posttraining)
    return dict(
        selections=len(report["selections"]),
        scopes=sorted(report["scopes"]),
        failures=report["failures"],
        passed=not report["failures"],
    )
