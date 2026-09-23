"""Búsqueda tabular secuencial con población común, intentos y resultados confirmados."""

import argparse
import copy
import fcntl
import importlib.metadata
import math
import os
import platform
from itertools import product
from pathlib import Path

import numpy as np
import torch

from mars_titan.budget_training import seed_run
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json, outside_source, sha256

from .checkpoints import StopRequest
from .external_corpus import run_external_reference
from .tabular_corpus import run_tabular_reference


def _code():
    root = Path(__file__).parents[1]
    names = (
        "training/tabular_search.py",
        "training/tabular_corpus.py",
        "training/external_corpus.py",
        "training/corpus_inputs.py",
        "training/cohort_contract.py",
        "training/checkpoints.py",
        "models/baselines/ridge.py",
        "models/baselines/boosting.py",
        "models/baselines/external_boosting.py",
        "models/baselines/inputs.py",
        "evaluation/session_metrics.py",
        "data/streaming.py",
        "data/batches.py",
        "data/storage.py",
        "data/cohort_files.py",
        "data/embeddings.py",
        "budget_training.py",
    )
    return {name: sha256(root / name) for name in names}


def _environment():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in {":4096:8", ":16:8"}:
        raise ValueError("Configura CUBLAS_WORKSPACE_CONFIG antes de iniciar la campaña")
    require_cuda()
    seed_run(0)
    return dict(
        python=platform.python_version(),
        numpy=np.__version__,
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        packages={
            name: importlib.metadata.version(name)
            for name in (
                "pyarrow",
                "scikit-learn",
                "xgboost",
                "cupy-cuda13x",
            )
        },
        cublas_workspace=os.environ["CUBLAS_WORKSPACE_CONFIG"],
    )


def _numbers(values, minimum, maximum, *, integers=False, count=8):
    return (
        isinstance(values, list)
        and 1 <= len(values) <= count
        and all(
            type(v) in ((int,) if integers else (int, float))
            and math.isfinite(v)
            and minimum <= v <= maximum
            for v in values
        )
        and len(set(values)) == len(values)
    )


def _xgb_case(parameters, stage):
    p = parameters
    rate = repr(float(p["learning_rate"]))
    name = f"xgb-d{p['max_depth']}-b{p['max_bin']}-lr{rate}-s{p['seed']}"
    return dict(id=name, kind="xgboost", stage=stage, parameters=p)


def _configuration(path):
    config, digest = read_manifest(path, 64 * 1024)
    keys = {
        "schema_version",
        "ridge_alphas",
        "depths",
        "bins",
        "rates",
        "rounds",
        "search_seed",
        "finalist_seeds",
        "batch_size",
        "max_batch_bytes",
        "max_host_cache_bytes",
        "on_host",
        "checkpoint_interval",
        "final_test_opened",
    }
    if (
        set(config) != keys
        or type(config["schema_version"]) is not int
        or config["schema_version"] != 1
        or not _numbers(config["ridge_alphas"], 1e-12, 1e12)
        or not _numbers(config["depths"], 1, 12, integers=True)
        or not _numbers(config["bins"], 2, 512, integers=True)
        or not _numbers(config["rates"], 1e-6, 1)
        or not _numbers(config["finalist_seeds"], 0, 2**31 - 1, integers=True)
        or type(config["search_seed"]) is not int
        or config["search_seed"] not in config["finalist_seeds"]
        or type(config["rounds"]) is not int
        or not 1 <= config["rounds"] <= 1000
        or type(config["batch_size"]) is not int
        or not 1 <= config["batch_size"] <= 4096
        or type(config["max_batch_bytes"]) is not int
        or not 1 <= config["max_batch_bytes"] <= 256 * 1024**2
        or type(config["max_host_cache_bytes"]) is not int
        or not 1 <= config["max_host_cache_bytes"] <= 16 * 1024**3
        or type(config["checkpoint_interval"]) is not int
        or not 1 <= config["checkpoint_interval"] <= config["rounds"]
        or type(config["on_host"]) is not bool
        or config["final_test_opened"] is not False
    ):
        raise ValueError("La configuración tabular no cumple el diseño o los presupuestos")
    common = {
        key: config[key]
        for key in (
            "rounds",
            "batch_size",
            "max_batch_bytes",
            "max_host_cache_bytes",
            "on_host",
            "checkpoint_interval",
        )
    }
    cases = [
        dict(
            id=f"ridge-a{float(alpha)!r}",
            kind="ridge",
            stage="search",
            parameters=dict(alpha=alpha, batch_size=config["batch_size"]),
        )
        for alpha in config["ridge_alphas"]
    ]
    grid = list(product(config["depths"], config["bins"], config["rates"]))
    if len(grid) > 12:
        raise ValueError("El diseño supera doce configuraciones de boosting")
    cases.extend(
        _xgb_case(
            dict(
                common,
                max_depth=depth,
                max_bin=bins,
                learning_rate=rate,
                seed=config["search_seed"],
            ),
            "search",
        )
        for depth, bins, rate in grid
    )
    if len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Los casos necesitan identificadores distintos")
    return config, cases, digest


def _artifact(folder, record):
    if not isinstance(record, dict) or not isinstance(record.get("path"), str):
        raise ValueError("Falta un artefacto confirmado")
    path = Path(record["path"])
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("El artefacto sale del directorio de la ejecución")
    path = folder / path
    safe_destination(path)
    if (
        not path.is_file()
        or path.stat().st_size > 4 * 1024**3
        or sha256(path) != record.get("sha256")
    ):
        raise ValueError("Un artefacto confirmado ha cambiado o falta")


def _completed(folder, task, source, source_hash):
    report, digest = read_manifest(folder / "run.json", 8 * 1024**2)
    identity = report.get("identity", {})
    kind = "ridge" if task["kind"] == "ridge" else "xgboost_external_cuda"
    if (
        report.get("status") != "completed"
        or report.get("model") != kind
        or identity.get("manifest_sha256", report.get("manifest_sha256")) != source_hash
        or report.get("scope") != source["scope"]
        or report.get("cohort_complete") != source["cohort_complete"]
        or report.get("samples") != source["counts"]
        or report.get("fitted_rows") != source["counts"]["train"]
        or report.get("final_test_opened") is not False
        or set(report.get("predictions", {})) != {"train", "validation"}
    ):
        raise ValueError("El resultado no confirma la misma población, origen y modelo")
    options = task["parameters"]
    if task["kind"] == "ridge":
        if any(report.get(key) != value for key, value in options.items()):
            raise ValueError("Ridge no conserva los parámetros del caso")
    elif identity.get("options") != options or report.get("completed_rounds") != options["rounds"]:
        raise ValueError("El boosting no conserva parámetros y rondas completas")
    _artifact(folder, report["checkpoint"])
    for partition, artifact in report["predictions"].items():
        _artifact(folder, artifact)
        if artifact["metrics"]["samples"] != source["counts"][partition]:
            raise ValueError("La evaluación no conserva todas las muestras")
    score = report["predictions"]["validation"]["metrics"]["session_mae"]
    if type(score) not in (int, float) or not math.isfinite(score) or score < 0:
        raise ValueError("El MAE por sesión no es finito y no negativo")
    return report, digest, score


class _Paused(Exception):
    """La campaña se detiene en una barrera recuperable."""


class _Study:
    def __init__(self, config_path, manifest, output, source, summary, stop):
        self.config_path, self.manifest, self.output = config_path, manifest, output
        self.source, self.summary, self.stop = source, summary, stop
        self.by_id = {item["id"]: item for item in summary["runs"]}
        self.visited = set()
        if len(self.by_id) != len(summary["runs"]):
            raise ValueError("Hay casos tabulares duplicados")

    def save(self):
        self.summary["completed_runs"] = sum(
            r["status"] == "completed" for r in self.by_id.values()
        )
        atomic_json(self.output / "summary.json", self.summary)

    def check_sources(self):
        identity = self.summary["identity"]
        if (
            sha256(self.config_path) != identity["config_sha256"]
            or sha256(self.manifest) != identity["manifest_sha256"]
            or _code() != identity["code"]
        ):
            raise ValueError("El diseño, origen o código cambió durante la campaña")

    def execute(self, task):
        self.visited.add(task["id"])
        if self.stop.requested:
            raise _Paused
        identity = self.summary["identity"]
        self.check_sources()
        item = self.by_id.get(task["id"])
        if item is None:
            item = dict(copy.deepcopy(task), status="pending", attempts=[])
            self.summary["runs"].append(item)
            self.by_id[item["id"]] = item
        if any(item.get(key) != value for key, value in task.items()):
            raise ValueError("Un caso tabular ha cambiado de configuración")
        attempt = item["attempts"][-1] if item["attempts"] else None
        folder = self.output / attempt["path"] if attempt else None
        if folder is not None:
            expected = f"runs/{task['id']}/attempt-{len(item['attempts']):04d}"
            if attempt["path"] != expected:
                raise ValueError("El intento no pertenece a la ruta prevista")
            safe_destination(folder)
            if item["status"] == "completed" or (
                (folder / "run.json").is_file()
                and read_manifest(folder / "run.json", 8 * 1024**2)[0].get("status") == "completed"
            ):
                report, digest, score = _completed(
                    folder, task, self.source, identity["manifest_sha256"]
                )
                if item["status"] == "completed" and (
                    item.get("report_sha256") != digest or item.get("session_mae") != score
                ):
                    raise ValueError("El informe o la puntuación tabular confirmados han cambiado")
                item.update(status="completed", report_sha256=digest, session_mae=score)
                attempt["status"] = "completed"
                self.save()
                return item
        resume = bool(task["kind"] == "xgboost" and folder and (folder / "run.json").is_file())
        if not resume:
            if len(item["attempts"]) >= 32:
                raise ValueError("Se ha alcanzado el límite de intentos del caso")
            if attempt and attempt["status"] == "running":
                attempt["status"] = "interrupted"
            attempt = dict(
                path=f"runs/{task['id']}/attempt-{len(item['attempts']) + 1:04d}", status="running"
            )
            item["attempts"].append(attempt)
            folder = self.output / attempt["path"]
        safe_destination(folder)
        item["status"] = attempt["status"] = "running"
        self.save()
        try:
            if task["kind"] == "ridge":
                report = run_tabular_reference(
                    self.manifest, folder, kind="ridge", **task["parameters"]
                )
            else:
                report = run_external_reference(
                    self.manifest, folder, resume=resume, stop=self.stop, **task["parameters"]
                )
            self.check_sources()
            if report["status"] == "paused":
                item["status"] = attempt["status"] = "paused"
                raise _Paused
            report, digest, score = _completed(
                folder, task, self.source, identity["manifest_sha256"]
            )
            item.update(status="completed", report_sha256=digest, session_mae=score)
            attempt["status"] = "completed"
        except _Paused:
            raise
        except BaseException as error:
            item["status"] = attempt["status"] = "failed"
            attempt["error"] = dict(type=type(error).__name__, message=str(error))
            raise
        finally:
            self.save()
        return item


def run_tabular_search(
    config_path, manifest, output, *, resume=False, stop=None, expected_source_hash=None
):
    config_path, manifest, output = map(Path, (config_path, manifest, output))
    config, cases, config_hash = _configuration(config_path)
    source, source_hash = read_manifest(manifest, 8 * 1024**2)
    if expected_source_hash is not None and source_hash != expected_source_hash:
        raise ValueError("El origen no coincide con la población verificada por la cola")
    if (
        source.get("kind") != "corpus_supervision"
        or source.get("scope") not in {"full_corpus", "development_snapshot"}
        or (source["scope"] == "full_corpus" and source.get("cohort_complete") is not True)
        or source.get("final_test_opened", False) is not False
        or set(source.get("counts", {})) != {"train", "validation"}
        or any(type(v) is not int or v < 1 for v in source["counts"].values())
    ):
        raise ValueError(
            "La campaña necesita dos particiones admitidas y una población identificada"
        )
    safe_destination(output)
    for root in (*source["roots"].values(), manifest.parent, config_path):
        outside_source(Path(root), output)
        outside_source(output, Path(root))
    if type(resume) is not bool or output.exists() and not resume or resume and not output.is_dir():
        raise ValueError("Usa una salida nueva o recuperación explícita de la campaña")
    identity = dict(
        config_sha256=config_hash,
        manifest_sha256=source_hash,
        code=_code(),
        environment=_environment(),
    )
    output.mkdir(parents=True, exist_ok=resume)
    descriptor = os.open(output / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        planned = len(cases) + len(config["finalist_seeds"]) - 1
        expected = dict(
            schema_version=1,
            identity=identity,
            planned_runs=planned,
            scope=source["scope"],
            cohort_complete=source["cohort_complete"],
            counts=source["counts"],
            final_test_opened=False,
        )
        marker = output / "initialization.json"
        if marker.exists():
            if read_manifest(marker)[0] != expected:
                raise ValueError("La inicialización pertenece a otra campaña")
        elif any(path.name != ".lock" for path in output.iterdir()):
            raise ValueError("La salida sin identidad contiene archivos ajenos o previos")
        else:
            atomic_json(marker, expected)
        confirmed = (output / "summary.json").is_file()
        if not confirmed and any(
            path.name not in {".lock", "initialization.json"} for path in output.iterdir()
        ):
            raise ValueError("Falta el resumen de una campaña que ya tiene artefactos")
        summary = (
            read_manifest(output / "summary.json")[0]
            if resume and confirmed
            else dict(
                schema_version=1,
                status="running",
                identity=identity,
                scope=source["scope"],
                cohort_complete=source["cohort_complete"],
                counts=source["counts"],
                planned_runs=planned,
                completed_runs=0,
                runs=[],
                selected={},
                final_test_opened=False,
            )
        )
        if any(
            type(summary.get(key)) is not type(value) or summary.get(key) != value
            for key, value in expected.items()
        ):
            raise ValueError("La campaña no corresponde al diseño, fuente o entorno actuales")
        if (
            summary.get("status") not in {"running", "paused", "failed", "completed"}
            or not isinstance(summary.get("runs"), list)
            or len(summary["runs"]) > planned
            or not all(
                isinstance(item, dict)
                and item.get("status") in {"running", "paused", "failed", "completed", "pending"}
                for item in summary["runs"]
            )
            or type(summary.get("completed_runs")) is not int
            or summary["completed_runs"]
            != sum(item["status"] == "completed" for item in summary["runs"])
            or not isinstance(summary.get("selected"), dict)
        ):
            raise ValueError("El resumen no conserva los estados y recuentos de la campaña")
        study = _Study(config_path, manifest, output, source, summary, stop or StopRequest())
        summary["status"] = "running"
        study.save()
        try:
            results = [study.execute(case) for case in cases]
            for kind in ("ridge", "xgboost"):
                winner = min(
                    (item for item in results if item["kind"] == kind),
                    key=lambda item: (item["session_mae"], item["id"]),
                )
                summary["selected"][kind] = winner["id"]
                if kind == "xgboost":
                    for seed in config["finalist_seeds"]:
                        if seed != config["search_seed"]:
                            study.execute(
                                _xgb_case(dict(winner["parameters"], seed=seed), "finalist")
                            )
            if set(study.by_id) != study.visited or summary["completed_runs"] != planned:
                raise ValueError("La campaña no reconcilia sus casos y resultados completos")
            study.check_sources()
            summary["status"] = "completed"
        except _Paused:
            summary["status"] = "paused"
        except BaseException as error:
            summary.update(
                status="failed", error=dict(type=type(error).__name__, message=str(error))
            )
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
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    with StopRequest() as stop:
        result = run_tabular_search(
            args.config, args.manifest, args.output, resume=args.resume, stop=stop
        )
    print(
        f"Estado: {result['status']}. "
        f"Casos terminados: {result['completed_runs']}/{result['planned_runs']}"
    )


if __name__ == "__main__":
    main()
