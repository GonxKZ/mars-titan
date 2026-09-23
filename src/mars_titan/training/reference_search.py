"""Búsqueda reproducible, finalistas y continuaciones sobre una edición congelada."""

import argparse
import fcntl
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.budget_training import seed_run
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json, outside_source, sha256

from .checkpoints import StopRequest
from .reference_campaign import _check_finished, campaign_views
from .reference_design import design_cases
from .reference_run import _confirmed_state, read_json, run_reference_case, scientific_identity


class _Paused(Exception):
    """Interrupción entre casos o en una barrera confirmada del entrenador."""


def _configuration(path):
    plan, digest = read_manifest(path, 64 * 1024)
    keys = {
        "schema_version",
        "scope",
        "arms",
        "models",
        "search_seed",
        "finalist_seeds",
        "max_epochs",
        "patience",
        "min_delta",
        "batch_size",
        "context_sessions",
        "checkpoint_seconds",
        "posttraining_epochs",
        "posttraining_learning_rate",
        "posttraining_losses",
        "pooled_weightings",
        "final_test_opened",
    }
    if (
        not isinstance(plan, dict)
        or set(plan) != keys
        or type(plan["schema_version"]) is not int
        or plan["schema_version"] != 1
        or plan["scope"] not in {"development_snapshot", "full_corpus"}
        or not isinstance(plan["arms"], list)
        or not plan["arms"]
        or len(set(plan["arms"])) != len(plan["arms"])
        or not set(plan["arms"]) <= {"US", "CN", "US+CN"}
        or type(plan["batch_size"]) is not int
        or not 1 <= plan["batch_size"] <= 4096
        or type(plan["context_sessions"]) is not int
        or plan["context_sessions"] not in {32, 64, 128}
        or type(plan["checkpoint_seconds"]) not in (int, float)
        or not math.isfinite(plan["checkpoint_seconds"])
        or not 0 < plan["checkpoint_seconds"] <= 900
        or type(plan["posttraining_epochs"]) is not int
        or not 1 <= plan["posttraining_epochs"] <= 30
        or type(plan["posttraining_learning_rate"]) not in (int, float)
        or not math.isfinite(plan["posttraining_learning_rate"])
        or not 0 < plan["posttraining_learning_rate"] <= 1
        or plan["posttraining_losses"] != ["mae", "mse"]
        or plan["pooled_weightings"] not in (["natural"], ["natural", "balanced_markets"])
        or plan["final_test_opened"] is not False
    ):
        raise ValueError("La configuración de búsqueda no cumple el contrato")
    seeds = plan["finalist_seeds"]
    if (
        not isinstance(seeds, list)
        or not 1 <= len(seeds) <= 8
        or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds)
        or len(set(seeds)) != len(seeds)
        or plan["search_seed"] not in seeds
    ):
        raise ValueError("La semilla de búsqueda debe ser uno de los finalistas, sin duplicados")
    cases = design_cases(
        plan["models"],
        seed=plan["search_seed"],
        epochs=plan["max_epochs"],
        patience=plan["patience"],
        min_delta=plan["min_delta"],
    )
    return plan, cases, digest


class _Study:
    """Confirmar casos identificados y volver a comprobar los que ya terminaron."""

    def __init__(self, output, plan, views, summary, stop, sources):
        self.output, self.plan, self.views, self.summary, self.stop = (
            output,
            plan,
            views,
            summary,
            stop,
        )
        self.by_id = {item["id"]: item for item in summary["runs"]}
        self.sources = sources
        self.visited = set()
        if len(self.by_id) != len(summary["runs"]):
            raise ValueError("Hay casos duplicados en la búsqueda")

    def save(self):
        self.summary["completed_runs"] = sum(
            r["status"] == "completed" for r in self.summary["runs"]
        )
        atomic_json(self.output / "summary.json", self.summary)

    def execute(self, task):
        self.visited.add(task["id"])
        if self.stop.requested:
            raise _Paused
        if any(sha256(path) != digest for path, digest in self.sources.items()):
            raise ValueError("La configuración o el manifiesto de origen han cambiado")
        if scientific_identity() != self.summary["identity"]["scientific"]:
            raise ValueError("El entorno o código científico ha cambiado")
        for name in ("reference_search.py", "reference_design.py"):
            if sha256(Path(__file__).with_name(name)) != self.summary["identity"]["code"][name]:
                raise ValueError("El diseño de búsqueda ha cambiado")
        view = self.output / "views" / f"{task['arm']}.json"
        if read_json(view) != self.views[task["arm"]]:
            raise ValueError("La población de un brazo ha cambiado")
        item = self.by_id.get(task["id"])
        if item is not None and any(item.get(key) != value for key, value in task.items()):
            raise ValueError("Un caso confirmado cambió de configuración u origen")
        if item is None:
            item = {**task, "status": "pending"}
            self.summary["runs"].append(item)
            self.by_id[item["id"]] = item
        folder = self.output / item["path"]
        safe_destination(folder)
        if item["status"] != "completed":
            parent = self.by_id.get(item.get("parent"))
            if "parent" in item and (parent is None or parent["status"] != "completed"):
                raise ValueError("La continuación necesita un origen terminado")
            item["status"] = "running"
            self.save()
            try:
                report = run_reference_case(
                    view,
                    folder,
                    item["case"],
                    batch_size=self.plan["batch_size"],
                    checkpoint_seconds=self.plan["checkpoint_seconds"],
                    resume=folder.exists(),
                    stop=self.stop,
                    weighting=item["weighting"],
                    initialize_from=self.output / parent["path"] if parent else None,
                )
                item.update(status=report["status"], report_sha256=sha256(folder / "run.json"))
                if report["status"] == "paused":
                    self.save()
                    raise _Paused
            except _Paused:
                raise
            except BaseException as error:
                item.update(status="failed", error_type=type(error).__name__, error=str(error))
                self.save()
                raise
        try:
            _check_finished(self.output, item, view)
            report = read_json(folder / "run.json")
            if (
                report["identity"]["case"] != task["case"]
                or report["identity"]["weighting"] != task["weighting"]
            ):
                raise ValueError("El informe no corresponde al caso y ponderación declarados")
            _confirmed_state(
                folder, report["identity"], report["checkpoint"], report.get("selection")
            )
            value = report["predictions"]["validation"]["metrics"]["session_mae"]
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError("El caso no tiene un MAE de sesión finito y no negativo")
            if "session_mae" in item and item["session_mae"] != value:
                raise ValueError("La puntuación registrada de un caso ha cambiado")
        except BaseException as error:
            item.update(status="failed", error_type=type(error).__name__, error=str(error))
            self.save()
            raise
        item["session_mae"] = value
        self.save()
        return item


def _task(arm, weighting, stage, identifier, case, parent=None):
    key = f"{arm}-{weighting}/{stage}/{identifier}"
    return dict(
        id=key,
        path=f"runs/{key}",
        arm=arm,
        weighting=weighting,
        stage=stage,
        case=case,
        **({"parent": parent} if parent else {}),
    )


def _execute_design(study, cases):
    plan, summary = study.plan, study.summary
    finalists, selected = [], {}
    for arm in plan["arms"]:
        for weighting in plan["pooled_weightings"] if arm == "US+CN" else ["natural"]:
            searches = [
                study.execute(_task(arm, weighting, "search", c["id"], c["case"])) for c in cases
            ]
            for kind in plan["models"]:
                winner = min(
                    (r for r in searches if r["case"]["kind"] == kind),
                    key=lambda r: (r["session_mae"], r["id"]),
                )
                selected[f"{arm}-{weighting}/{kind}"] = winner["id"]
                summary["selected"] = dict(selected)
                study.save()
                for seed in plan["finalist_seeds"]:
                    finalist = (
                        winner
                        if seed == plan["search_seed"]
                        else study.execute(
                            _task(
                                arm,
                                weighting,
                                "finalist",
                                f"{kind}-s{seed}",
                                {**winner["case"], "seed": seed},
                            )
                        )
                    )
                    finalists.append(
                        dict(
                            arm=arm,
                            weighting=weighting,
                            kind=kind,
                            seed=seed,
                            run_id=finalist["id"],
                        )
                    )
                    summary["finalists"] = list(finalists)
                    study.save()
                    for loss in plan["posttraining_losses"]:
                        case = {k: v for k, v in finalist["case"].items() if k != "selection"}
                        case.update(
                            loss=loss,
                            epochs=plan["posttraining_epochs"],
                            learning_rate=plan["posttraining_learning_rate"],
                        )
                        study.execute(
                            _task(
                                arm,
                                weighting,
                                "posttraining",
                                f"{kind}-{loss}-s{seed}",
                                case,
                                finalist["id"],
                            )
                        )
    if set(study.by_id) != study.visited:
        raise ValueError("El registro contiene casos que no pertenecen al diseño")
    summary.update(finalists=finalists, selected=selected)


def run_search(config: Path, manifest: Path, output: Path, *, resume=False):
    plan, cases, config_hash = _configuration(config)
    views = campaign_views(manifest, plan["arms"])
    first = next(iter(views.values()))
    if first["context_sessions"] != plan["context_sessions"]:
        raise ValueError("El contexto de la edición no coincide con el diseño de búsqueda")
    source_hash = first["source_manifest_sha256"]
    if sha256(manifest) != source_hash or sha256(config) != config_hash:
        raise ValueError("La configuración o el origen han cambiado durante la lectura")
    if first["scope"] != plan["scope"] or (
        plan["scope"] == "full_corpus" and not first["cohort_complete"]
    ):
        raise ValueError("La edición no acredita el alcance de corpus completo solicitado")
    for source in (*(Path(p) for p in first["roots"].values()), Path("dataset")):
        outside_source(source, output)
        outside_source(output, source)
    safe_destination(output)
    if (output.exists() and not resume) or (resume and not output.is_dir()):
        raise ValueError("Usa un estudio nuevo o solicita continuar uno existente")
    arms = sum(len(plan["pooled_weightings"]) if arm == "US+CN" else 1 for arm in plan["arms"])
    planned = (
        arms
        * len(plan["models"])
        * (12 + len(plan["finalist_seeds"]) - 1 + len(plan["finalist_seeds"]) * 2)
    )
    if planned > 512:
        raise ValueError("La búsqueda supera el presupuesto de 512 ejecuciones")
    require_cuda()
    seed_run(0)
    identity = dict(
        config_sha256=config_hash,
        manifest_sha256=source_hash,
        configuration=plan,
        scientific=scientific_identity(),
        code={
            name: sha256(Path(__file__).with_name(name))
            for name in ("reference_search.py", "reference_design.py")
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = (
            read_json(output / "summary.json")
            if resume
            else dict(
                schema_version=1,
                kind="reference_search",
                identity=identity,
                status="running",
                scope=first["scope"],
                cohort_complete=first["cohort_complete"],
                runs=[],
                selected={},
                finalists=[],
                planned_runs=planned,
                completed_runs=0,
                final_test_opened=False,
                started_at_utc=datetime.now(UTC).isoformat(),
            )
        )
        expected = dict(
            schema_version=1,
            kind="reference_search",
            identity=identity,
            scope=first["scope"],
            cohort_complete=first["cohort_complete"],
            planned_runs=planned,
            final_test_opened=False,
        )
        if any(
            type(summary.get(key)) is not type(value) or summary.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("La identidad o el contrato del estudio han cambiado")
        if (
            summary.get("status") not in {"running", "paused", "failed", "completed"}
            or not isinstance(summary.get("runs"), list)
            or not isinstance(summary.get("selected"), dict)
            or not isinstance(summary.get("finalists"), list)
        ):
            raise ValueError("El contrato del registro de búsqueda no es válido")
        for arm, view in views.items():
            path = output / "views" / f"{arm}.json"
            safe_destination(path)
            if path.exists():
                if read_json(path) != view:
                    raise ValueError("El brazo corresponde a otra población")
            else:
                atomic_json(path, view)
        with StopRequest() as stop:
            study = _Study(
                output, plan, views, summary, stop, {config: config_hash, manifest: source_hash}
            )
            summary["status"] = "running"
            study.save()
            try:
                _execute_design(study, cases)
                if len(summary["runs"]) != planned or any(
                    r["status"] != "completed" for r in summary["runs"]
                ):
                    raise ValueError("La búsqueda no reconcilia todas sus ejecuciones")
                summary["status"] = "completed"
            except _Paused:
                summary["status"] = "paused"
            except BaseException as error:
                summary.update(status="failed", error_type=type(error).__name__, error=str(error))
                raise
            finally:
                study.save()
        return summary
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continuar casos pendientes")
    args = parser.parse_args()
    result = run_search(args.config, args.manifest, args.output, resume=args.resume)
    print(
        json.dumps(
            {k: result[k] for k in ("status", "completed_runs", "planned_runs")}, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
