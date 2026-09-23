"""Campaña recuperable de tres controles predictivos con un padre congelado común."""

import argparse
import fcntl
import os
import time
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256

from .checkpoints import StopRequest
from .predictive_inputs import PredictiveDataset, fit_standardizer
from .predictive_parents import prepare_parent_cache
from .predictive_run import _code, _options, run_predictive_case


def _configuration(path):
    config, digest = read_manifest(path, 64 * 1024)
    if (
        set(config)
        != {
            "schema_version",
            "modes",
            "seeds",
            "epochs",
            "learning_rate",
            "weight_decay",
            "clip_norm",
            "batch_size",
            "checkpoint_seconds",
            "final_test_opened",
        }
        or config["schema_version"] != 1
        or config["modes"] != ["reinforce", "expected", "mae"]
        or not isinstance(config["seeds"], list)
        or not 1 <= len(config["seeds"]) <= 8
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in config["seeds"])
        or len(config["seeds"]) != len(set(config["seeds"]))
        or config["final_test_opened"] is not False
    ):
        raise ValueError("El diseño necesita los tres controles, semillas únicas y test cerrado")
    cases = []
    for seed in config["seeds"]:
        for mode in config["modes"]:
            case = dict(
                mode=mode,
                seed=seed,
                **{
                    key: config[key]
                    for key in ("epochs", "learning_rate", "weight_decay", "clip_norm")
                },
            )
            _options(case, config["batch_size"], 0, config["checkpoint_seconds"])
            cases.append(dict(id=f"{mode}-s{seed}", case=case, path=f"runs/{mode}-s{seed}"))
    return config, cases, digest


def run_predictive_study(config_path, ordered, parent, output, *, resume=False, stop=None):
    config_path, ordered, parent, output = map(Path, (config_path, ordered, parent, output))
    config, cases, config_hash = _configuration(config_path)
    source, source_hash = read_manifest(ordered)
    _, parent_hash = read_manifest(parent, 8 * 1024**2)
    identity = dict(
        config_sha256=config_hash,
        ordered_sha256=source_hash,
        parent_report_sha256=parent_hash,
        code=_code() | {"predictive_study.py": sha256(Path(__file__))},
    )
    safe_destination(output)
    for protected in (ordered.parent, parent.parent):
        outside_source(protected, output)
        outside_source(output, protected)
    if output.exists() and not resume or resume and not (output / "summary.json").is_file():
        raise ValueError("La campaña requiere una salida nueva o recuperación explícita")
    output.mkdir(parents=True, exist_ok=resume)
    descriptor = os.open(output / ".study.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    started, stop = time.perf_counter(), stop or StopRequest()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        summary = (
            read_manifest(output / "summary.json")[0]
            if resume
            else dict(
                schema_version=1,
                status="running",
                identity=identity,
                planned_runs=len(cases),
                completed_runs=0,
                scope=source["scope"],
                cohort_complete=source["cohort_complete"],
                counts=source["counts"],
                final_test_opened=False,
                runs=[],
            )
        )
        if summary["identity"] != identity or summary["planned_runs"] != len(cases):
            raise ValueError("La campaña pertenece a otro diseño, fuente, padre o código")
        summary["status"] = "running"
        atomic_json(output / "summary.json", summary)
        try:
            cache = output / "parent-cache"
            prepare_parent_cache(ordered, parent, cache, resume=cache.exists())
            summary["parent_cache_sha256"] = sha256(cache / "manifest.json")
            normalization = output / "normalization.json"
            if not normalization.exists():
                with PredictiveDataset(ordered, cache / "manifest.json") as dataset:
                    atomic_json(
                        normalization, fit_standardizer(dataset, batch_size=config["batch_size"])
                    )
            digest = sha256(normalization)
            if "normalization_sha256" in summary and summary["normalization_sha256"] != digest:
                raise ValueError("La normalización compartida ha cambiado")
            summary["normalization_sha256"] = digest
            atomic_json(output / "summary.json", summary)
            by_id = {item["id"]: item for item in summary["runs"]}
            if len(by_id) != len(summary["runs"]):
                raise ValueError("La campaña contiene casos duplicados")
            for task in cases:
                if stop.requested:
                    summary["status"] = "paused"
                    break
                if read_manifest(config_path, 64 * 1024)[1] != config_hash or _code() != {
                    key: value
                    for key, value in identity["code"].items()
                    if key != "predictive_study.py"
                }:
                    raise ValueError("El diseño o el código ha cambiado durante la campaña")
                item = by_id.get(task["id"])
                if item is None:
                    item = dict(task, status="pending")
                    summary["runs"].append(item)
                    by_id[task["id"]] = item
                if any(item.get(key) != value for key, value in task.items()):
                    raise ValueError("Un caso confirmado cambió de configuración")
                folder = output / task["path"]
                safe_destination(folder)
                if (
                    item["status"] == "completed"
                    and sha256(folder / "run.json") != item["report_sha256"]
                ):
                    raise ValueError("El informe confirmado de un control ha cambiado")
                item["status"] = "running"
                atomic_json(output / "summary.json", summary)
                report = run_predictive_case(
                    ordered,
                    cache / "manifest.json",
                    folder,
                    task["case"],
                    batch_size=config["batch_size"],
                    checkpoint_seconds=config["checkpoint_seconds"],
                    resume=folder.exists(),
                    normalization=normalization,
                    stop=stop,
                )
                item.update(
                    status=report["status"],
                    report_sha256=sha256(folder / "run.json"),
                    global_step=report["global_step"],
                )
                if report["status"] == "completed":
                    item["session_mae"] = report["predictions"]["validation"]["metrics"]["median"][
                        "session_mae"
                    ]
                summary["completed_runs"] = sum(
                    item["status"] == "completed" for item in summary["runs"]
                )
                atomic_json(output / "summary.json", summary)
                if report["status"] == "paused":
                    summary["status"] = "paused"
                    break
            else:
                summary["status"] = "completed"
            if summary["status"] == "completed" and summary["completed_runs"] != len(cases):
                raise ValueError("La campaña no ha completado todos sus controles")
        except BaseException as error:
            summary.update(
                status="failed", last_failure=dict(type=type(error).__name__, message=str(error))
            )
            raise
        finally:
            summary["attempt_seconds"] = time.perf_counter() - started
            atomic_json(output / "summary.json", summary)
        return summary
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "ordered", "parent", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = vars(parser.parse_args())
    args["config_path"] = args.pop("config")
    with StopRequest() as stop:
        result = run_predictive_study(**args, stop=stop)
    print(
        f"Estado: {result['status']}. "
        f"Controles terminados: {result['completed_runs']}/{result['planned_runs']}"
    )


if __name__ == "__main__":
    main()
