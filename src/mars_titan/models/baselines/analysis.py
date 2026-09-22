"""Comparar predicciones finales de una campaña verificada, sin cargar modelos."""

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from itertools import product
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.storage import sha256
from mars_titan.models.baselines.diagnostics import paired_date_bootstrap, point_diagnostics

_MAX_JSON_BYTES = 5 * 1024**2
_MAX_ROWS = 100_000
_PARTITIONS = ("train", "validation")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"El JSON contiene una clave duplicada: {key}")
        result[key] = value
    return result


def _read_json(path):
    with path.open("rb") as stream:
        encoded = stream.read(_MAX_JSON_BYTES + 1)
    if len(encoded) > _MAX_JSON_BYTES:
        raise ValueError("El JSON supera el límite de 5 MiB")
    value = json.loads(encoded, object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("El JSON debe contener un objeto")
    return value


def _digest(value):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError("La huella SHA-256 no es válida")
    return value


def _artifact(root, item, name):
    relative = Path(item[f"{name}_path"])
    path = (root / relative).resolve()
    if relative.is_absolute() or ".." in relative.parts or not path.is_relative_to(root):
        raise ValueError(f"La ruta de {name} sale de la campaña")
    if not path.is_file() or path.stat().st_size > 128 * 1024**2:
        raise ValueError(f"El artefacto {name} falta o supera el límite de 128 MiB")
    if sha256(path) != _digest(item[f"{name}_sha256"]):
        raise ValueError(f"La huella de {name} no coincide con el recibo")
    return path


def _run_key(item):
    return item["stage"], item["kind"], item["loss"], item["learning_rate"], item["seed"]


def _matches(actual, expected):
    return all(
        type(actual.get(key)) is type(value) and actual[key] == value
        for key, value in expected.items()
    )


def _check_campaign(summary):
    runs, plan = summary["runs"], summary["resolved_config"]
    if not isinstance(runs, list) or not 1 <= len(runs) <= 128:
        raise ValueError("La campaña necesita entre una y 128 ejecuciones")
    if not _matches(
        summary,
        {
            "schema_version": 1,
            "status": "completed",
            "final_test_opened": False,
            "planned_runs": len(runs),
            "completed_runs": len(runs),
            "failed_runs": 0,
        },
    ) or not _matches(
        plan,
        {"training_through": "2022-12-31", "validation_year": 2023, "final_test_opened": False},
    ):
        raise ValueError("El recibo no acredita una campaña completa con las particiones acordadas")
    _digest(summary["config_sha256"])
    _check_grid(runs, plan)
    by_id = {item["id"]: item for item in runs}
    for item in runs:
        _check_run(item, by_id, plan)


def _check_grid(runs, plan):
    validators = {
        "models": lambda value: (
            isinstance(value, str) and value in {"rnn", "lstm", "gru", "dlinear"}
        ),
        "seeds": lambda value: type(value) is int and 0 <= value < 2**32,
        "losses": lambda value: isinstance(value, str) and value in {"mse", "mae", "huber"},
        "learning_rates": lambda value: (
            type(value) in (int, float) and 0 < value <= sys.float_info.max and math.isfinite(value)
        ),
    }
    for name, valid in validators.items():
        values = plan[name]
        if (
            not isinstance(values, list)
            or not values
            or not all(valid(value) for value in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(
                f"La rejilla requiere {name} válidos, sin valores ausentes ni duplicados"
            )
    post = plan["posttraining"]
    if type(plan["epochs"]) is not int or not 2 <= plan["epochs"] <= 30:
        raise ValueError("La rejilla requiere entre 2 y 30 épocas base")
    if (
        not isinstance(post, dict)
        or not _matches(
            post,
            {"base_loss": "mse", "base_learning_rate": 0.001, "learning_rate": 0.0001, "epochs": 5},
        )
        or not isinstance(post.get("losses"), list)
        or len(post["losses"]) != 2
        or any(type(value) is not str for value in post["losses"])
        or set(post["losses"]) != {"mae", "mse"}
        or post["base_loss"] not in plan["losses"]
        or post["base_learning_rate"] not in plan["learning_rates"]
    ):
        raise ValueError("El postentrenamiento no conserva sus controles y su origen declarados")
    cases = (
        len(plan["models"])
        * len(plan["seeds"])
        * (len(plan["losses"]) * len(plan["learning_rates"]) + len(post["losses"]))
    )
    if cases > 128:
        raise ValueError("La rejilla supera el presupuesto de 128 casos, incluidos los controles")
    expected = {
        ("base", kind, loss, rate, seed)
        for kind, loss, rate, seed in product(
            plan["models"], plan["losses"], plan["learning_rates"], plan["seeds"]
        )
    } | {
        ("posttraining", kind, loss, post["learning_rate"], seed)
        for kind, loss, seed in product(plan["models"], post["losses"], plan["seeds"])
    }
    if (
        len({row["id"] for row in runs}) != len(runs)
        or len(expected) != len(runs)
        or {_run_key(row) for row in runs} != expected
    ):
        raise ValueError("Las identidades de las ejecuciones no coinciden con la rejilla declarada")


def _check_run(item, by_id, plan):
    if (
        not isinstance(item["id"], str)
        or not item["id"]
        or item["kind"] not in {"rnn", "lstm", "gru", "dlinear"}
        or item["loss"] not in {"mse", "mae", "huber"}
        or type(item["seed"]) is not int
        or not 0 <= item["seed"] < 2**32
        or type(item["epochs"]) is not int
        or item["epochs"] < 1
        or type(item["learning_rate"]) not in (float, int)
        or not math.isfinite(item["learning_rate"])
        or item["learning_rate"] <= 0
        or not _matches(item, {"status": "completed", "final_test_opened": False})
    ):
        raise ValueError("Una ejecución tiene una identidad o un estado no válido")
    if item["stage"] == "base":
        if "parent" in item or item["epochs"] != plan["epochs"]:
            raise ValueError("Una ejecución base tiene un origen o unas épocas incorrectos")
        return
    parent, post = by_id[item["parent"]], plan["posttraining"]
    expected_parent = (
        "base",
        item["kind"],
        post["base_loss"],
        post["base_learning_rate"],
        item["seed"],
    )
    if (
        _run_key(parent) != expected_parent
        or item["epochs"] != post["epochs"]
        or item["parent_checkpoint_path"] != parent["checkpoint_path"]
        or item["parent_checkpoint_sha256"] != parent["checkpoint_sha256"]
    ):
        raise ValueError("El postentrenamiento no comparte el padre y la semilla declarados")


def _data_hashes(report):
    normalized = {}
    for name, digest in report["input_hashes"].items():
        if name.startswith("src/"):
            continue
        parts = Path(name).parts
        if "targets" in parts:
            name = Path(*parts[parts.index("targets") :]).as_posix()
        if name in normalized:
            raise ValueError("Hay una colisión entre rutas normalizadas de una misma fuente")
        normalized[name] = _digest(digest)
    if not normalized:
        raise ValueError("El informe no contiene huellas de datos")
    return normalized


def _verified_report(root, item, base_epochs):
    paths = {
        name: _artifact(root, item, name)
        for name in ("report", "checkpoint", "predictions", "training_predictions")
    }
    report = _read_json(paths["report"])
    config = report["config"]
    if (
        report["status"] != "completed"
        or report["model"] != item["kind"]
        or report["final_test_opened"] is not False
        or report["training_cutoff"] != "2022-12-31"
        or report["validation_year"] != 2023
        or report["restored_predictions_equal"] is not True
        or report["resume_check"]["exact_weights"] is not True
        or report["samples"] != item["samples"]
        or report["epochs"][-1]["epoch"] != item["epochs"]
        or report["epochs"][-1]["checkpoint_sha256"] != item["checkpoint_sha256"]
        or any(config[key] != item[key] for key in ("kind", "loss", "seed", "epochs"))
        or config["lr"] != item["learning_rate"]
        or any(
            report[f"{name}_sha256"] != item[f"{name}_sha256"]
            for name in ("predictions", "training_predictions")
        )
    ):
        raise ValueError(
            "El informe no coincide con la identidad, las huellas o la recuperación del recibo"
        )
    initialization = config["initialization"]
    if item["stage"] == "posttraining":
        if (
            not isinstance(initialization, dict)
            or initialization["sha256"] != item["parent_checkpoint_sha256"]
            or initialization["policy"] != "weights_only_new_optimizer_and_rng"
            or initialization["source_completed_epochs"] != base_epochs
            or initialization != item["initialization"]
        ):
            raise ValueError("El informe no acredita la inicialización desde el padre declarado")
    elif initialization is not None:
        raise ValueError("La ejecución base declara una inicialización inesperada")
    return report, paths


def _predictions(path, kind, partition, expected_n):
    columns = ["asset_id", "prediction_at", "target", kind, "zero"]
    with pq.ParquetFile(path) as parquet:
        schema = parquet.schema_arrow
        if (
            not set(columns) <= set(schema.names)
            or len(set(schema.names)) != len(schema.names)
            or type(expected_n) is not int
            or not 1 <= expected_n <= _MAX_ROWS
            or parquet.metadata.num_rows != expected_n
        ):
            raise ValueError("Las columnas o las filas del Parquet no coinciden con el contrato")
        timestamp = schema.field("prediction_at").type
        if not pa.types.is_timestamp(timestamp) or timestamp.tz is None:
            raise ValueError("Las fechas deben ser timestamps con zona horaria")
        if not (
            pa.types.is_string(schema.field("asset_id").type)
            or pa.types.is_large_string(schema.field("asset_id").type)
        ):
            raise ValueError("asset_id debe contener identificadores de texto")
        if any(
            not (
                pa.types.is_floating(schema.field(name).type)
                or pa.types.is_integer(schema.field(name).type)
            )
            for name in columns[2:]
        ):
            raise ValueError("Los objetivos y las predicciones deben ser numéricos")
        frame = parquet.read(columns=columns, use_threads=False).to_pandas()
    if frame.isna().any().any() or frame.asset_id.str.strip().eq("").any():
        raise ValueError("Las predicciones contienen valores ausentes o identidades vacías")
    if (
        not np.isfinite(frame[columns[2:]].to_numpy(dtype=np.float64)).all()
        or frame.zero.ne(0).any()
    ):
        raise ValueError("Las predicciones deben ser finitas y la referencia zero debe valer cero")
    frame["prediction_at"] = frame.prediction_at.dt.tz_convert("UTC")
    frame["date"] = frame.prediction_at.dt.strftime("%Y-%m-%d")
    if frame.duplicated(["asset_id", "date"]).any():
        raise ValueError("La población contiene más de una predicción por activo y fecha")
    years = frame.prediction_at.dt.year
    if (partition == "train" and years.gt(2022).any()) or (
        partition == "validation" and years.ne(2023).any()
    ):
        raise ValueError("Las fechas de predicción salen de la partición declarada")
    return frame.sort_values(["asset_id", "prediction_at"], kind="stable").reset_index(drop=True)


def _check_metrics(report, partition, kind, metrics, target):
    key = "final_training_row_metrics" if partition == "train" else "diagnostic_row_metrics"
    expected = report[key]
    zero = point_diagnostics(target, np.zeros(len(target)))
    for name, actual in ((kind, metrics), ("zero", zero)):
        for metric in ("mae", "mse"):
            value = expected[name][metric]
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or not math.isclose(value, actual[metric], rel_tol=1e-9, abs_tol=1e-12)
            ):
                raise ValueError(f"El {metric} de {partition} no coincide con los pesos finales")


def _describe(values):
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _cross_sectional(frame, kind):
    pearson, spearman = [], []
    insufficient, constant = 0, 0
    for _, group in frame.groupby("date", sort=True):
        if group.asset_id.nunique() < 3:
            insufficient += 1
        elif group.target.nunique() < 2 or group[kind].nunique() < 2:
            constant += 1
        else:
            metrics = point_diagnostics(group.target.to_numpy(), group[kind].to_numpy())
            pearson.append(metrics["pearson"])
            spearman.append(metrics["spearman"])
    return {
        "minimum_assets": 3,
        "eligible_dates_n": len(pearson),
        "excluded_insufficient_assets_n": insufficient,
        "excluded_constant_n": constant,
        "pearson": _describe(pearson) if pearson else None,
        "spearman": _describe(spearman) if spearman else None,
        "reason": None
        if pearson
        else "No hay fechas con tres activos y ambas series no constantes",
        "aggregation": "equal_date_descriptive_not_independent_observations",
    }


def _breakdown(frame, kind, groups):
    result = {}
    for name, group in frame.groupby(groups, sort=True):
        metrics = point_diagnostics(group.target.to_numpy(), group[kind].to_numpy())
        result[name] = {
            key: metrics[key] for key in ("n", "mae", "rmse", "bias", "q95_absolute_error")
        }
    return result


def _seed_groups(runs):
    grouped = {}
    for run in runs:
        key = (run["stage"], run["kind"], run["loss"], run["learning_rate"])
        grouped.setdefault(key, []).append(run)
    result = []
    for (stage, kind, loss, rate), group in sorted(grouped.items()):
        result.append(
            {
                "stage": stage,
                "kind": kind,
                "loss": loss,
                "learning_rate": rate,
                "seeds": sorted(row["seed"] for row in group),
                "n_seeds": len(group),
                "range_kind": "observed_seed_range_not_confidence_interval",
                "std_convention": "sample_ddof_1_null_for_one_seed",
                **{
                    partition: {
                        metric: _describe([row[partition]["metrics"][metric] for row in group])
                        for metric in ("mae", "mse", "rmse", "bias")
                    }
                    for partition in _PARTITIONS
                },
            }
        )
    return result


def _resources(report, item):
    phases = [epoch[partition] for epoch in report["epochs"] for partition in _PARTITIONS]
    rss = [
        phase["sampled_tree_peak"]["rss_mib"] for phase in phases if "sampled_tree_peak" in phase
    ]
    epochs = {}
    for partition in _PARTITIONS:
        readings = [epoch[partition] for epoch in report["epochs"]]
        complete = len(readings) == item["epochs"] and all(
            "elapsed_seconds" in phase and "samples" in phase for phase in readings
        )
        seconds = sum(phase["elapsed_seconds"] for phase in readings) if complete else None
        samples = sum(phase["samples"] for phase in readings) if complete else None
        epochs[partition] = {
            "elapsed_seconds_total": seconds,
            "processed_samples": samples,
            "samples_per_second": samples / seconds if seconds else None,
            "complete_epoch_timing": complete,
            "latency_scope": "loader_plus_transfer_plus_synchronized_gpu_step",
            "step_latency_by_epoch": [
                {
                    "epoch": epoch["epoch"],
                    **{f"p{q}_ms": epoch[partition][f"step_p{q}_ms"] for q in (50, 95, 99)},
                }
                for epoch in report["epochs"]
                if all(f"step_p{q}_ms" in epoch[partition] for q in (50, 95, 99))
            ],
        }
    return {
        "process_lifetime_peak_rss_mib": report.get("peak_rss_mib"),
        "sampled_run_peak_rss_mib": max(rss) if rss else None,
        "ram_poll_seconds": sorted(
            {phase["ram_poll_seconds"] for phase in phases if "ram_poll_seconds" in phase}
        ),
        "sampled_memory_scope": "training_and_validation_epochs_process_tree",
        "sampled_memory_limit": "El muestreo no garantiza el máximo ni cubre toda la ejecución",
        "process_peak_limit": "El máximo histórico puede proceder de otra ejecución del proceso",
        "runner_wall_seconds": report.get("total_seconds"),
        "campaign_case_wall_seconds": item.get("elapsed_seconds"),
        "runner_timing_scope": (
            "preparation_training_validation_checkpoint_replay_and_final_predictions"
        ),
        "epochs": epochs,
        "kernel_seconds": None,
        "kernel_seconds_reason": "El paso medido no aísla el tiempo de los kernels",
    }


def _posttraining_pairs(items, predictions, population, bootstrap_options):
    grouped, by_id = {}, {item["id"]: item for item in items}
    for item in items:
        if item["stage"] == "posttraining":
            grouped.setdefault(item["parent"], {})[item["loss"]] = item["id"]
    result = []
    for parent, controls in sorted(grouped.items()):
        source = by_id[parent]
        pair = {
            "parent": parent,
            "kind": source["kind"],
            "seed": source["seed"],
            "parent_checkpoint_sha256": source["checkpoint_sha256"],
            "post_mae": controls["mae"],
            "post_mse": controls["mse"],
        }
        for partition in _PARTITIONS:
            frame = population[partition]
            pair[partition] = {}
            for name, model_id, reference_id in (
                ("mae_vs_mse", controls["mae"], controls["mse"]),
                ("mae_vs_parent", controls["mae"], parent),
                ("mse_vs_parent", controls["mse"], parent),
            ):
                predicted, reference = (
                    predictions[model_id][partition],
                    predictions[reference_id][partition],
                )
                comparison = point_diagnostics(frame.target.to_numpy(), predicted, reference)[
                    "reference"
                ]
                if partition == "validation":
                    comparison["bootstrap"] = paired_date_bootstrap(
                        frame.date.to_numpy(),
                        frame.target.to_numpy(),
                        predicted,
                        reference,
                        **bootstrap_options,
                    )
                pair[partition][name] = comparison
        result.append(pair)
    return result


def _encoded(result):
    encoded = (json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    if len(encoded) >= _MAX_JSON_BYTES:
        raise ValueError("El análisis supera el límite de 5 MiB, no se ha escrito una salida")
    return encoded


def analyze_campaign(campaign: Path, *, block_length=5, repetitions=2000, seed=42) -> dict:
    """Validar y analizar una cohorte completa sin leer pesos ni recalibrar etiquetas.

    Mantiene una población común y un vector de predicción por ejecución y
    partición. Limita la campaña a 128 ejecuciones y 100 000 filas por ejecución.
    Los desgloses son descriptivos y los intervalos remuestrean fechas completas.
    Las semillas describen variación del entrenamiento sobre los mismos datos.
    """
    campaign = Path(campaign).resolve()
    root, summary = campaign.parent, _read_json(campaign)
    try:
        _check_campaign(summary)
        bootstrap_options = {"block_length": block_length, "repetitions": repetitions, "seed": seed}
        runs, predictions, population = [], {}, {}
        common_hashes = None
        for item in summary["runs"]:
            report, paths = _verified_report(root, item, summary["resolved_config"]["epochs"])
            hashes = _data_hashes(report)
            if common_hashes is not None and hashes != common_hashes:
                raise ValueError(
                    "Las fuentes de datos no coinciden entre ejecuciones de la cohorte"
                )
            common_hashes = hashes
            if "data_provenance" in summary and summary["data_provenance"] != {
                "input_hashes": hashes,
                "samples": report["samples"],
            }:
                raise ValueError("La procedencia declarada no coincide con las fuentes verificadas")
            if sum(report["samples"][partition] for partition in _PARTITIONS) > _MAX_ROWS:
                raise ValueError("Una ejecución supera el límite de 100 000 filas")
            run = {
                key: item[key]
                for key in ("id", "kind", "loss", "learning_rate", "seed", "stage", "epochs")
            }
            run["numerics"] = report["numerics"]
            run["resources"] = _resources(report, item)
            run["artifacts"] = {
                name: {"path": item[f"{name}_path"], "sha256": item[f"{name}_sha256"]}
                for name in paths
            }
            predictions[item["id"]] = {}
            for partition in _PARTITIONS:
                name = "training_predictions" if partition == "train" else "predictions"
                frame = _predictions(
                    paths[name], item["kind"], partition, report["samples"][partition]
                )
                identity = frame[["asset_id", "prediction_at", "target", "zero", "date"]]
                if partition in population and not identity.equals(population[partition]):
                    raise ValueError("La población o los objetivos no coinciden entre ejecuciones")
                population.setdefault(partition, identity.copy())
                predicted = frame[item["kind"]].to_numpy(dtype=np.float64, copy=True)
                target = frame.target.to_numpy(dtype=np.float64)
                predictions[item["id"]][partition] = predicted
                metrics = point_diagnostics(target, predicted)
                _check_metrics(report, partition, item["kind"], metrics, target)
                run[partition] = {
                    "metrics": metrics,
                    "by_asset": _breakdown(frame, item["kind"], "asset_id"),
                    "by_month": _breakdown(
                        frame, item["kind"], frame.prediction_at.dt.strftime("%Y-%m")
                    ),
                    "cross_sectional": _cross_sectional(frame, item["kind"]),
                }
                if partition == "validation":
                    run[partition]["bootstrap_vs_zero"] = paired_date_bootstrap(
                        frame.date.to_numpy(), target, predicted, **bootstrap_options
                    )
            runs.append(run)
        result = {
            "schema_version": 1,
            "campaign_sha256": sha256(campaign),
            "config_sha256": summary["config_sha256"],
            "analysis_sha256": sha256(Path(__file__)),
            "diagnostics_sha256": sha256(Path(point_diagnostics.__code__.co_filename)),
            "data_fingerprint": hashlib.sha256(
                json.dumps(sorted(common_hashes.items())).encode()
            ).hexdigest(),
            "cohort_policy": "one_campaign_common_population",
            "final_test_opened": False,
            "population": {
                partition: {
                    "n": len(frame),
                    "n_dates": int(frame.date.nunique()),
                    "assets": sorted(frame.asset_id.unique().tolist()),
                    "first_date": frame.date.min(),
                    "last_date": frame.date.max(),
                }
                for partition, frame in population.items()
            },
            "runs": runs,
            "seed_groups": _seed_groups(runs),
            "posttraining_pairs": _posttraining_pairs(
                summary["runs"], predictions, population, bootstrap_options
            ),
            "not_computed": {
                "percentage_errors": {
                    "value": None,
                    "reason": "MAPE no es estable con retornos nulos o cercanos a cero",
                },
                "probabilistic_scores": {
                    "value": None,
                    "reason": "Las salidas son puntuales y no tienen una distribución predictiva",
                },
                "calibration": {
                    "value": None,
                    "reason": "No hay probabilidades ni intervalos predictivos calibrados",
                },
                "financial_metrics": {
                    "value": None,
                    "reason": "No hay una estrategia con posiciones, costes y ejecución definidos",
                },
            },
            "limitations": [
                "La cohorte es una muestra dirigida y la validación no es el test final",
                "El error de entrenamiento usa los pesos finales sobre los datos de ajuste",
                "Activos y meses son desgloses descriptivos, no repeticiones independientes",
                "Los intervalos por bloques son exploratorios, sin corrección por selección",
                "El rango entre semillas no es un intervalo de confianza sobre periodos nuevos",
                "La correlación agrupada difiere de la correlación transversal diaria",
                "Se conservan las etiquetas maduras, sin volver a estimar beta ni el índice",
                "No se acredita aquí el corte temporal del preentrenamiento de los codificadores",
                "Las huellas comprueban integridad y coherencia, no la autenticidad de los datos",
            ],
        }
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError(f"El contrato de campaña o informe está incompleto: {error}") from error
    _encoded(result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-length", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output
    project = Path(point_diagnostics.__code__.co_filename).resolve().parents[4]
    summary = _read_json(campaign)
    protected = [campaign.parent, project / "dataset"]
    for name in ("prepared", "samples"):
        if name in summary.get("inputs", {}):
            protected.append((campaign.parent / summary["inputs"][name]).resolve())
    if output.exists() or output.is_symlink():
        raise ValueError("Usa una salida nueva para conservar los archivos existentes")
    if any(output.resolve().is_relative_to(path) for path in protected):
        raise ValueError("La salida debe estar fuera de la campaña y de sus datos de origen")
    result = analyze_campaign(
        campaign, block_length=args.block_length, repetitions=args.repetitions, seed=args.seed
    )
    encoded = _encoded(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.") as temporary:
        temporary.write(encoded)
        temporary.flush()
        os.fsync(temporary.fileno())
        try:
            os.link(temporary.name, output)
        except FileExistsError as error:
            raise ValueError(
                "Usa una salida nueva para conservar los archivos existentes"
            ) from error
    print(f"Análisis guardado de {len(result['runs'])} ejecuciones")


if __name__ == "__main__":
    main()
