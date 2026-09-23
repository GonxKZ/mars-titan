"""Campañas recuperables sobre ediciones explícitas, separadas de las sondas."""

import argparse
import fcntl
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.budget_training import seed_run
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.campaign import _cases, _load_config

from .checkpoints import StopRequest
from .cohort_contract import cohort_identity
from .corpus_inputs import CorpusDataset
from .reference_run import read_json, run_reference_case, scientific_identity


def campaign_views(manifest: Path, arms: list[str]) -> dict:
    """Mantener la unión exacta de activos y sus particiones, sin sustituir mercados."""
    source = CorpusDataset(manifest)
    views = {}
    for arm in arms:
        markets = {"US", "CN"} if arm == "US+CN" else {arm}
        assets = [a for a in source.assets if a["market"] in markets]
        if any(
            sum(a["counts"][p] for a in assets if a["market"] == market) == 0
            for market in markets
            for p in ("train", "validation")
        ):
            raise ValueError(f"Falta población de ajuste o validación para el brazo {arm}")
        views[arm] = {
            **source.manifest,
            "assets": assets,
            "counts": {p: sum(a["counts"][p] for a in assets) for p in ("train", "validation")},
            "selected_arm": arm,
            "source_manifest_sha256": source.identity,
        }
        if source.cohort:
            coverage = [row for row in source.manifest["coverage"] if row["market"] in markets]
            views[arm].update(
                coverage=coverage,
                candidate_count=len(coverage),
                markets=sorted(markets),
                samples=sum(row["samples"] for row in coverage if row["state"] == "encoded"),
                failed_assets=sum(row["state"] == "failed" for row in coverage),
            )
            cohort_identity(views[arm])
    return views


def _configuration(path):
    value = read_json(path)
    keys = {
        "schema_version",
        "scope",
        "arms",
        "recipe",
        "batch_size",
        "checkpoint_seconds",
        "pooled_weightings",
        "final_test_opened",
    }
    if (
        set(value) != keys
        or value["schema_version"] != 2
        or value["scope"] not in {"development_snapshot", "full_corpus"}
        or not isinstance(value["arms"], list)
        or not value["arms"]
        or len(set(value["arms"])) != len(value["arms"])
        or not set(value["arms"]) <= {"US", "CN", "US+CN"}
        or type(value["batch_size"]) is not int
        or not 1 <= value["batch_size"] <= 4096
        or type(value["checkpoint_seconds"]) not in (int, float)
        or not 0 < value["checkpoint_seconds"] <= 900
        or value["pooled_weightings"] not in (["natural"], ["natural", "balanced_markets"])
        or value["final_test_opened"] is not False
    ):
        raise ValueError("La configuración no cumple el contrato de la campaña 2")
    return value, (path.parent / value["recipe"]).resolve()


def _check_finished(output, item, view):
    path = output / item["path"] / "run.json"
    if sha256(path) != item["report_sha256"]:
        raise ValueError("El recibo de una configuración terminada ha cambiado")
    report = read_json(path)
    if (
        report["status"] != "completed"
        or report["final_test_opened"] is not False
        or report["identity"]["manifest_sha256"] != sha256(view)
    ):
        raise ValueError("El recibo no acredita la población y cierre esperados")
    for record in (*report["predictions"].values(), report["checkpoint"]):
        if sha256(path.parent / record["path"]) != record["sha256"]:
            raise ValueError("Ha cambiado un artefacto de una configuración terminada")


def run_reference_campaign(
    config: Path, manifest: Path, output: Path, *, resume: bool = False
) -> dict:
    """Ejecutar los casos pendientes y conservar recibos terminados sin reentrenarlos."""
    plan, recipe_path = _configuration(config)
    recipe = _load_config(recipe_path)
    views = campaign_views(manifest, plan["arms"])
    meta = next(iter(views.values()))
    if meta["scope"] != plan["scope"] or (
        plan["scope"] == "full_corpus" and not meta["cohort_complete"]
    ):
        raise ValueError("El alcance de la edición no acredita el corpus completo solicitado")
    for protected in (*(Path(p) for p in meta["roots"].values()), Path("dataset")):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.is_symlink() or (output.exists() and not resume) or (resume and not output.is_dir()):
        raise ValueError("Usa una campaña nueva o solicita continuar una existente")
    require_cuda()
    seed_run(0)
    identity = dict(
        config_sha256=sha256(config),
        recipe_sha256=sha256(recipe_path),
        manifest_sha256=sha256(manifest),
        scientific=scientific_identity(),
    )
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("La campaña ya tiene un proceso activo") from error
        summary_path = output / "summary.json"
        if resume:
            summary = read_json(summary_path)
            if summary["identity"] != identity:
                raise ValueError("La identidad de datos, código o configuración ha cambiado")
        else:
            runs = []
            for arm in plan["arms"]:
                for weighting in plan["pooled_weightings"] if arm == "US+CN" else ["natural"]:
                    prefix = f"{arm}-{weighting}"
                    for case in _cases(recipe):
                        item = dict(
                            id=f"{prefix}/{case['id']}",
                            path=f"runs/{prefix}/{case['id']}",
                            arm=arm,
                            weighting=weighting,
                            status="pending",
                            case={
                                **{
                                    k: case[k]
                                    for k in ("kind", "loss", "learning_rate", "seed", "epochs")
                                },
                                "huber_delta": recipe["huber_delta"],
                            },
                        )
                        if "parent" in case:
                            item["parent"] = f"{prefix}/{case['parent']}"
                        runs.append(item)
            summary = dict(
                schema_version=2,
                identity=identity,
                scope=plan["scope"],
                cohort_complete=meta["cohort_complete"],
                status="running",
                runs=runs,
                planned_runs=len(runs),
                completed_runs=0,
                final_test_opened=False,
                started_at_utc=datetime.now(UTC).isoformat(),
            )
            for arm, value in views.items():
                atomic_json(output / "views" / f"{arm}.json", value)
            atomic_json(summary_path, summary)
        by_id = {r["id"]: r for r in summary["runs"]}
        with StopRequest() as stop:
            for item in summary["runs"]:
                if scientific_identity() != identity["scientific"]:
                    raise ValueError("La identidad científica ha cambiado durante la campaña")
                view = output / "views" / f"{item['arm']}.json"
                if read_json(view) != views[item["arm"]]:
                    raise ValueError("La población del brazo ha cambiado")
                if item["status"] == "completed":
                    _check_finished(output, item, view)
                    continue
                if stop.requested:
                    summary["status"] = "paused"
                    atomic_json(summary_path, summary)
                    return summary
                folder = output / item["path"]
                parent = output / by_id[item["parent"]]["path"] if "parent" in item else None
                if parent is not None and by_id[item["parent"]]["status"] != "completed":
                    raise ValueError("La continuación requiere un origen ya terminado")
                item["status"] = "running"
                summary["status"] = "running"
                atomic_json(summary_path, summary)
                try:
                    report = run_reference_case(
                        view,
                        folder,
                        item["case"],
                        batch_size=plan["batch_size"],
                        checkpoint_seconds=plan["checkpoint_seconds"],
                        resume=folder.exists(),
                        initialize_from=parent,
                        stop=stop,
                        weighting=item["weighting"],
                    )
                    item["status"] = report["status"]
                    item["report_sha256"] = sha256(folder / "run.json")
                    if report["status"] == "completed":
                        _check_finished(output, item, view)
                        summary["completed_runs"] += 1
                    elif report["status"] == "paused":
                        summary["status"] = "paused"
                        return summary
                    else:
                        raise ValueError("La referencia terminó sin un estado verificable")
                except BaseException as error:
                    item.update(status="failed", error_type=type(error).__name__, error=str(error))
                    summary["status"] = "failed"
                    raise
                finally:
                    atomic_json(summary_path, summary)
            summary.update(status="completed", finished_at_utc=datetime.now(UTC).isoformat())
            atomic_json(summary_path, summary)
            return summary
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "manifest", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true", help="Continuar los casos pendientes")
    args = parser.parse_args()
    result = run_reference_campaign(args.config, args.manifest, args.output, resume=args.resume)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))


if __name__ == "__main__":
    main()
