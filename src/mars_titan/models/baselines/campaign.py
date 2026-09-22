"""Ejecutar rejillas acotadas de referencias y controles de postentrenamiento."""

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from itertools import product
from pathlib import Path

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.gru_probe import run_temporal_probe

_FIXED = {
    "schema_version": 1,
    "batch_size": 16,
    "workers": 0,
    "device": "cuda:0",
    "training_through": "2022-12-31",
    "validation_year": 2023,
    "final_test_opened": False,
}
_POST_FIXED = {
    "base_loss": "mse",
    "base_learning_rate": 0.001,
    "learning_rate": 0.0001,
    "epochs": 5,
    "selection": "fixed_control_not_validation_winner",
    "optimizer_policy": "new_adamw",
}


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"La configuración contiene una clave duplicada: {key}")
        result[key] = value
    return result


def _positive(value):
    return type(value) in (int, float) and 0 < value <= sys.float_info.max and math.isfinite(value)


def _values(plan, key, valid):
    values = plan[key]
    if (
        not isinstance(values, list)
        or not values
        or not all(valid(value) for value in values)
        or len(set(values)) != len(values)
    ):
        raise ValueError(f"{key} requiere una lista válida, no vacía y sin duplicados")
    return values


def _fixed(plan, expected):
    for key, value in expected.items():
        if type(plan.get(key)) is not type(value) or plan[key] != value:
            raise ValueError(f"La campaña debe conservar {key}={value}")


def _load_config(path: Path) -> dict:
    with path.open("rb") as stream:
        encoded = stream.read(64 * 1024 + 1)
    if len(encoded) > 64 * 1024:
        raise ValueError("La configuración supera el presupuesto de 64 KiB")
    plan = json.loads(encoded, object_pairs_hook=_unique_object)
    keys = set(_FIXED) | {
        "purpose",
        "models",
        "seeds",
        "losses",
        "learning_rates",
        "epochs",
        "huber_delta",
        "posttraining",
    }
    if not isinstance(plan, dict) or set(plan) != keys:
        raise ValueError("La configuración no contiene las claves requeridas del esquema 1")
    _fixed(plan, _FIXED)
    if not isinstance(plan["purpose"], str) or not plan["purpose"].strip():
        raise ValueError("La campaña necesita un propósito no vacío")
    _values(
        plan,
        "models",
        lambda value: isinstance(value, str) and value in {"rnn", "lstm", "gru", "dlinear"},
    )
    _values(plan, "seeds", lambda value: type(value) is int and 0 <= value < 2**32)
    _values(
        plan, "losses", lambda value: isinstance(value, str) and value in {"mse", "mae", "huber"}
    )
    plan["learning_rates"] = [float(value) for value in _values(plan, "learning_rates", _positive)]
    if len(set(plan["learning_rates"])) != len(plan["learning_rates"]):
        raise ValueError("Las tasas de aprendizaje producen casos duplicados al convertirlas")
    if type(plan["epochs"]) is not int or not 2 <= plan["epochs"] <= 30:
        raise ValueError("La campaña requiere entre 2 y 30 épocas base")
    if not _positive(plan["huber_delta"]):
        raise ValueError("huber_delta debe ser finito y positivo")
    post = plan["posttraining"]
    if not isinstance(post, dict) or set(post) != set(_POST_FIXED) | {"losses"}:
        raise ValueError("El postentrenamiento no contiene las claves requeridas")
    _fixed(post, _POST_FIXED)
    _values(post, "losses", lambda value: isinstance(value, str) and value in {"mae", "mse"})
    if set(post["losses"]) != {"mae", "mse"}:
        raise ValueError("El postentrenamiento requiere los controles emparejados MAE y MSE")
    if (
        post["base_loss"] not in plan["losses"]
        or post["base_learning_rate"] not in plan["learning_rates"]
    ):
        raise ValueError("La rejilla debe incluir el origen MSE con tasa 0.001")
    cases = (
        len(plan["models"])
        * len(plan["seeds"])
        * (len(plan["losses"]) * len(plan["learning_rates"]) + len(post["losses"]))
    )
    if cases > 128:
        raise ValueError("La campaña supera el presupuesto de 128 casos, incluidos los controles")
    return plan


def _case(kind, loss, rate, seed, epochs, *, parent=None):
    identifier = f"{kind}-post-{loss}-s{seed}" if parent else f"{kind}-{loss}-{rate}-s{seed}"
    return {
        "id": identifier,
        "kind": kind,
        "loss": loss,
        "learning_rate": rate,
        "seed": seed,
        "epochs": epochs,
        "stage": "posttraining" if parent else "base",
        **({"parent": parent} if parent else {}),
        "output_path": f"runs/{identifier}",
        "report_path": f"reports/{identifier}.json",
        "checkpoint_path": f"runs/{identifier}/epoch-{epochs}.pt",
        "predictions_path": f"runs/{identifier}/predictions.parquet",
        "status": "pending",
    }


def _cases(plan):
    cases = [
        _case(kind, loss, rate, seed, plan["epochs"])
        for kind, loss, rate, seed in product(
            plan["models"], plan["losses"], plan["learning_rates"], plan["seeds"]
        )
    ]
    post = plan["posttraining"]
    for kind, seed, loss in product(plan["models"], plan["seeds"], post["losses"]):
        parent = f"{kind}-{post['base_loss']}-{post['base_learning_rate']}-s{seed}"
        cases.append(_case(kind, loss, post["learning_rate"], seed, post["epochs"], parent=parent))
    return cases


def _code_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[4],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def _record_report(item, report, output):
    if report.get("status") != "completed" or report.get("final_test_opened") is not False:
        raise ValueError(
            "El informe no acredita una ejecución completada con el test final cerrado"
        )
    last_epoch = report["epochs"][-1]
    if last_epoch["epoch"] != item["epochs"]:
        raise ValueError("El informe no acredita la última época acordada")
    for name, expected in (
        ("checkpoint", last_epoch["checkpoint_sha256"]),
        ("predictions", report["predictions_sha256"]),
    ):
        digest = sha256(output / item[f"{name}_path"])
        if digest != expected:
            raise ValueError(f"La huella de {name} no coincide con el informe")
        item[f"{name}_sha256"] = digest
    if "training_predictions_sha256" in report:
        item["training_predictions_path"] = f"{item['output_path']}/training-predictions.parquet"
        digest = sha256(output / item["training_predictions_path"])
        if digest != report["training_predictions_sha256"]:
            raise ValueError("La huella de las predicciones de entrenamiento no coincide")
        item["training_predictions_sha256"] = digest
    if "parent" in item:
        initialization = report["config"]["initialization"]
        if initialization["sha256"] != item["parent_checkpoint_sha256"]:
            raise ValueError("La inicialización no corresponde al punto de control padre")
        item["initialization"] = initialization
    item["report_sha256"] = sha256(output / item["report_path"])
    item["final_test_opened"] = report["final_test_opened"]
    for key in ("samples", "parameters", "restored_predictions_equal", "numerics"):
        if key in report:
            item[key] = report[key]
    for original, target in (
        ("diagnostic_row_metrics", "metrics"),
        ("final_training_row_metrics", "final_training_row_metrics"),
    ):
        if original in report:
            item[target] = report[original]
    if "exact_weights" in report.get("resume_check", {}):
        item["exact_recovery"] = report["resume_check"]["exact_weights"]


def run_campaign(config: Path, prepared: Path, samples: Path, output: Path) -> dict:
    """Ejecutar una campaña nueva y detenerla conservando el primer fallo."""
    plan = _load_config(config)
    for source in (
        prepared,
        samples,
        Path("dataset"),
        Path(__file__).resolve().parents[4] / "dataset",
    ):
        outside_source(source, output)
    if output.exists() or output.is_symlink():
        raise ValueError("Usa un directorio nuevo para conservar las campañas existentes")
    if not prepared.is_dir() or not samples.is_dir():
        raise ValueError("Se requieren directorios existentes de datos preparados y muestras")
    cases = _cases(plan)
    for item in cases:
        item["options"] = {
            key: item[key] for key in ("kind", "loss", "learning_rate", "seed", "epochs")
        }
        item["options"]["huber_delta"] = plan["huber_delta"]
        if "parent" in item:
            path = f"runs/{item['parent']}/epoch-{plan['epochs']}.pt"
            item["parent_checkpoint_path"] = path
            item["options"]["initialize_from"] = path
    started = time.perf_counter()
    summary = {
        "schema_version": 1,
        "purpose": plan["purpose"],
        "resolved_config": plan,
        "config_sha256": sha256(config),
        "code_commit": _code_commit(),
        "code_sha256": sha256(Path(__file__)),
        "inputs": {
            name: Path(os.path.relpath(path.resolve(), output.resolve())).as_posix()
            for name, path in (("config", config), ("prepared", prepared), ("samples", samples))
        },
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "planned_runs": len(cases),
        "completed_runs": 0,
        "failed_runs": 0,
        "final_test_opened": False,
        "runs": cases,
    }
    output.mkdir(parents=True, exist_ok=False)
    summary_path = output / "summary.json"
    atomic_json(summary_path, summary)
    by_id = {item["id"]: item for item in cases}
    for item in cases:
        item["status"] = "running"
        atomic_json(summary_path, summary)
        case_started = time.perf_counter()
        try:
            options = dict(item["options"])
            if "parent" in item:
                checkpoint = output / item["parent_checkpoint_path"]
                digest = sha256(checkpoint)
                if digest != by_id[item["parent"]]["checkpoint_sha256"]:
                    raise ValueError("El punto de control padre ha cambiado desde su registro")
                item["parent_checkpoint_sha256"] = digest
                options["initialize_from"] = checkpoint
            report = run_temporal_probe(
                prepared,
                samples,
                output / item["output_path"],
                output / item["report_path"],
                **options,
            )
            if report.get("final_test_opened") is not False:
                summary["final_test_opened"] = (
                    True if report.get("final_test_opened") is True else None
                )
            _record_report(item, report, output)
            item["status"] = "completed"
            summary["completed_runs"] += 1
        except BaseException as error:
            item.update(status="failed", error_type=type(error).__name__, error=str(error))
            summary.update(
                status="failed", failed_runs=1, finished_at_utc=datetime.now(UTC).isoformat()
            )
            raise
        finally:
            item["elapsed_seconds"] = time.perf_counter() - case_started
            summary["elapsed_seconds"] = time.perf_counter() - started
            atomic_json(summary_path, summary)
    summary.update(status="completed", finished_at_utc=datetime.now(UTC).isoformat())
    atomic_json(summary_path, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("config", "prepared", "samples", "output"):
        parser.add_argument(f"--{option}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_campaign(args.config, args.prepared, args.samples, args.output), indent=2))


if __name__ == "__main__":
    main()
