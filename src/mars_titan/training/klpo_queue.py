"""Admitir los seis padres terminados antes de comparar sus ajustes predictivos."""

import argparse
import fcntl
import math
import os
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.corpus_source import prepare_causal_corpus

from .baseline_queue import reference_view
from .checkpoints import StopRequest
from .predictive_run import _code
from .predictive_study import _configuration, run_predictive_study
from .run_receipts import initialize_receipt
from .tabular_search import _artifact, _completed


def _summary(path):
    summary, digest = read_manifest(path, 8 * 1024**2)
    if (
        summary.get("status") != "completed"
        or summary.get("scope") != "full_corpus"
        or summary.get("cohort_complete") is not True
        or summary.get("final_test_opened") is not False
        or type(summary.get("planned_runs")) is not int
        or not 1 <= summary["planned_runs"] <= 512
        or summary.get("completed_runs") != summary["planned_runs"]
        or not isinstance(summary.get("runs"), list)
        or len(summary["runs"]) != summary["planned_runs"]
        or not isinstance(summary.get("selected"), dict)
        or any(
            not isinstance(item, dict)
            or not isinstance(item.get("id"), str)
            or item.get("status") != "completed"
            or type(item.get("session_mae")) not in (int, float)
            or not math.isfinite(item["session_mae"])
            or item["session_mae"] < 0
            for item in summary["runs"]
        )
        or len({item["id"] for item in summary["runs"]}) != len(summary["runs"])
    ):
        raise ValueError("La cola necesita una campaña completa, única y sin test final")
    return summary, digest


def _winner(summary, candidates, key):
    if not candidates:
        raise ValueError("Falta una familia de referencias en la búsqueda")
    chosen = min(candidates, key=lambda item: (item["session_mae"], item["id"]))
    if summary["selected"].get(key) != chosen["id"]:
        raise ValueError("La referencia seleccionada no concuerda con la búsqueda confirmada")
    return chosen


def selected_parents(reference, tabular, arm="US"):
    """Verificar la población común y seleccionar por validación, nunca por el test."""
    reference, tabular = Path(reference), Path(tabular)
    proof = reference_view(reference, arm)
    neural, neural_hash = _summary(reference)
    tables, table_hash = _summary(tabular)
    source, source_hash = read_manifest(Path(proof["manifest"]), 8 * 1024**2)
    if (
        tables["identity"].get("manifest_sha256") != source_hash
        or tables.get("counts") != source["counts"]
        or neural_hash != proof["reference_summary_sha256"]
    ):
        raise ValueError("Los padres tabulares y neuronales no comparten la misma población")
    parents = {}
    for kind in ("rnn", "lstm", "gru", "dlinear"):
        candidates = [
            item
            for item in neural["runs"]
            if item.get("arm") == arm
            and item.get("weighting") == "natural"
            and item.get("stage") == "search"
            and item.get("case", {}).get("kind") == kind
        ]
        chosen = _winner(neural, candidates, f"{arm}-natural/{kind}")
        for item in candidates:
            path = reference.parent / item["path"] / "run.json"
            report, digest = read_manifest(path, 8 * 1024**2)
            if (
                digest != item["report_sha256"]
                or report["identity"].get("case", {}).get("kind") != kind
                or report["predictions"]["validation"]["metrics"]["session_mae"]
                != item["session_mae"]
            ):
                raise ValueError("Un padre neuronal no conserva su familia o puntuación")
        path = reference.parent / chosen["path"] / "run.json"
        parents[kind] = dict(
            report=str(path.resolve()), sha256=chosen["report_sha256"], run_id=chosen["id"]
        )
    for item in tables["runs"]:
        attempts = item.get("attempts")
        if not isinstance(attempts, list) or not 1 <= len(attempts) <= 32:
            raise ValueError("Falta un intento tabular confirmado")
        attempt = attempts[-1]
        expected = f"runs/{item['id']}/attempt-{len(attempts):04d}"
        if attempt.get("path") != expected or attempt.get("status") != "completed":
            raise ValueError("La ruta del intento tabular no corresponde a su caso")
        _artifact(tabular.parent, dict(path=f"{expected}/run.json", sha256=item["report_sha256"]))
        _, digest, score = _completed(tabular.parent / expected, item, source, source_hash)
        if digest != item["report_sha256"] or score != item["session_mae"]:
            raise ValueError("Un resultado tabular confirmado ha cambiado")
    for kind in ("ridge", "xgboost"):
        candidates = [
            item
            for item in tables["runs"]
            if item.get("kind") == kind and item.get("stage") == "search"
        ]
        chosen = _winner(tables, candidates, kind)
        path = tabular.parent / chosen["attempts"][-1]["path"] / "run.json"
        parents[kind] = dict(
            report=str(path.resolve()), sha256=chosen["report_sha256"], run_id=chosen["id"]
        )
    if (
        sha256(reference) != neural_hash
        or sha256(tabular) != table_hash
        or sha256(Path(proof["manifest"])) != source_hash
    ):
        raise ValueError("Una campaña ha cambiado durante la admisión de sus padres")
    return proof | dict(tabular_summary_sha256=table_hash, parents=parents)


def _queue_code():
    return _code() | {
        f"training/{name}": sha256(Path(__file__).with_name(name))
        for name in (
            "klpo_queue.py",
            "predictive_study.py",
            "baseline_queue.py",
            "tabular_search.py",
        )
    }


def _finished_study(folder, expected, cases):
    """Comprobar los artefactos terminados sin reescribir sus recibos."""
    result, digest = read_manifest(folder / "summary.json", 8 * 1024**2)
    if (
        digest != expected["sha256"]
        or result.get("status") != "completed"
        or result.get("completed_runs") != len(cases)
        or len(result.get("runs", [])) != len(cases)
    ):
        raise ValueError("El estudio terminado ha cambiado o está incompleto")
    by_id = {item["id"]: item for item in result["runs"]}
    if len(by_id) != len(cases):
        raise ValueError("El estudio terminado contiene casos duplicados")
    for case in cases:
        item = by_id.get(case["id"])
        if (
            item is None
            or any(item.get(key) != value for key, value in case.items())
            or item["status"] != "completed"
        ):
            raise ValueError("Un ajuste terminado no corresponde al diseño")
        path = folder / item["path"]
        _artifact(folder, dict(path=f"{item['path']}/run.json", sha256=item["report_sha256"]))
        report, _ = read_manifest(path / "run.json", 8 * 1024**2)
        if report.get("status") != "completed" or report.get("final_test_opened") is not False:
            raise ValueError("Un ajuste confirmado no está terminado o ha abierto el test")
        _artifact(path, report["checkpoint"])
        for artifact in report["predictions"].values():
            _artifact(path, artifact)
    return result


def run_queue(config, reference, tabular, output, *, arm="US", stop=None):
    """Preparar una única copia ordenada y recorrer todos los controles de cada padre."""
    config, reference, tabular, output = map(Path, (config, reference, tabular, output))
    plan, cases, config_hash = _configuration(config)
    if plan["schema_version"] != 2:
        raise ValueError("La cola requiere el diseño KLPO con sus controles emparejados")
    safe_destination(output)
    proof = selected_parents(reference, tabular, arm)
    for protected in (reference.parent, tabular.parent, Path(proof["manifest"]).parent, config):
        outside_source(protected, output)
        outside_source(output, protected)
    identity = dict(proof=proof, config_sha256=config_hash, code=_queue_code())
    summary_path = output / "summary.json"
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".queue.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    stop = stop or StopRequest()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(output, identity, record="summary.json", lock=".queue.lock")
        summary = (
            read_manifest(summary_path, 8 * 1024**2)[0]
            if confirmed
            else dict(
                schema_version=1,
                status="running",
                identity=identity,
                parents={},
                planned_parents=len(proof["parents"]),
                planned_runs=len(cases) * len(proof["parents"]),
                completed_parents=0,
                completed_runs=0,
                counts=proof["counts"],
                final_test_opened=False,
            )
        )
        expected = dict(
            schema_version=1,
            identity=identity,
            planned_parents=len(proof["parents"]),
            planned_runs=len(cases) * len(proof["parents"]),
            counts=proof["counts"],
            final_test_opened=False,
        )
        if any(
            type(summary.get(key)) is not type(value) or summary.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("La cola pertenece a otra identidad de datos, padres o código")
        if not isinstance(summary.get("parents"), dict) or not set(summary["parents"]) <= set(
            proof["parents"]
        ):
            raise ValueError("La cola contiene padres ajenos al diseño")
        summary["status"] = "running"
        atomic_json(summary_path, summary)
        try:
            ordered = output / "ordered"
            prepared = prepare_causal_corpus(
                Path(proof["manifest"]),
                ordered,
                batch_size=plan["batch_size"],
                resume=ordered.exists(),
                stop=stop,
            )
            if prepared["status"] == "paused":
                summary["status"] = "paused"
                return summary
            ordered_hash = sha256(ordered / "manifest.json")
            if "ordered_sha256" in summary and summary["ordered_sha256"] != ordered_hash:
                raise ValueError("El corpus ordenado de la cola ha cambiado")
            summary["ordered_sha256"] = ordered_hash
            for kind, parent in proof["parents"].items():
                if stop.requested:
                    summary["status"] = "paused"
                    break
                if sha256(config) != config_hash or _queue_code() != identity["code"]:
                    raise ValueError("El diseño o el código de la cola ha cambiado")
                if sha256(Path(parent["report"])) != parent["sha256"]:
                    raise ValueError("El informe de un padre ha cambiado tras su admisión")
                folder = output / "parents" / kind
                safe_destination(folder)
                previous = summary["parents"].get(kind)
                if previous is not None and previous["status"] == "completed":
                    result = _finished_study(folder, previous, cases)
                else:
                    result = run_predictive_study(
                        config,
                        ordered / "manifest.json",
                        Path(parent["report"]),
                        folder,
                        resume=folder.exists(),
                        stop=stop,
                    )
                if sha256(config) != config_hash or _queue_code() != identity["code"]:
                    raise ValueError(
                        "El diseño o el código de la cola ha cambiado durante el padre"
                    )
                if result["counts"] != proof["counts"] or result["planned_runs"] != len(cases):
                    raise ValueError("El ajuste no conserva la población o todos sus controles")
                summary["parents"][kind] = dict(
                    status=result["status"],
                    completed_runs=result["completed_runs"],
                    sha256=sha256(folder / "summary.json"),
                    path=str((folder / "summary.json").relative_to(output)),
                )
                summary["completed_parents"] = sum(
                    item["status"] == "completed" for item in summary["parents"].values()
                )
                summary["completed_runs"] = sum(
                    item["completed_runs"] for item in summary["parents"].values()
                )
                atomic_json(summary_path, summary)
                if result["status"] == "paused":
                    summary["status"] = "paused"
                    break
            else:
                if sha256(config) != config_hash or _queue_code() != identity["code"]:
                    raise ValueError("La identidad ha cambiado antes de confirmar la cola")
                summary["status"] = "completed"
        except BaseException as error:
            summary.update(
                status="failed", last_failure=dict(type=type(error).__name__, message=str(error))
            )
            raise
        finally:
            atomic_json(summary_path, summary)
        return summary
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "reference", "tabular", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=("US", "CN", "US+CN"), default="US")
    args = vars(parser.parse_args())
    with StopRequest() as stop:
        result = run_queue(**args, stop=stop)
    print(
        f"Estado: {result['status']}. "
        f"Ajustes terminados: {result['completed_runs']}/{result['planned_runs']}"
    )


if __name__ == "__main__":
    main()
