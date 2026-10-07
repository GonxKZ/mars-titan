"""Evaluar selecciones congeladas en calibración y evaluación, sin abrir el test final."""

import argparse
import fcntl
import gc
import os
import resource
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

from mars_titan.data.batches import atomic_parquet_batches
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.environments.actions import ActionGrid
from mars_titan.evaluation.session_metrics import SessionErrors
from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.models.predictive_adaptation import LinearResidualPolicy, gaussian_log_probabilities
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.klpo_queue import selected_parents
from mars_titan.training.predictive_parents import _verified_file
from mars_titan.training.run_receipts import initialize_receipt

from .evaluation import centers
from .parent_selection import matching_parents, matching_seeds, parent_for_seed
from .parents import load_parent
from .run import _best_state, code_identity

PARTITIONS = ("calibration", "evaluation")


def evaluate_partition(
    dataset,
    parent,
    partition,
    destination,
    *,
    stop,
    device,
    model=None,
    grid=None,
    neural=False,
    batch_size=256,
    predictor=None,
):
    """Recorrer un bloque posterior con el estado ya seleccionado y acumuladores acotados."""
    if partition not in PARTITIONS or dataset.temporal is None:
        raise ValueError("Solo se admiten calibración y evaluación de una vista temporal")
    if (
        type(batch_size) is not int
        or not 1 <= batch_size <= 256
        or (model is None) != (grid is None)
        or (model is not None and predictor is not None)
    ):
        raise ValueError("El lote o el contrato del modelo congelado no son válidos")
    names = ("prediction", "parent", "zero") + (("center",) if model is not None else ())
    errors = {name: SessionErrors() for name in names}
    if model is not None:
        model.eval().requires_grad_(False)
        values = torch.tensor(grid.values, dtype=torch.float64, device=device)

    def tables():
        with torch.inference_mode():
            for batch in dataset.batches(
                partition=partition, batch_size=batch_size, epoch=0, seed=0
            ):
                if stop.requested:
                    raise InterruptedError("Evaluación interrumpida antes de confirmar el bloque")
                inherited = parent.predict(batch["inputs"])
                predictions = dict(
                    prediction=inherited, parent=inherited, zero=np.zeros(len(inherited))
                )
                if predictor is not None:
                    predictions["prediction"] = predictor.predict(batch["inputs"])
                if model is not None:
                    features = np.concatenate(
                        [batch["inputs"][k].reshape(len(inherited), -1) for k in MODALITIES]
                        + [inherited[:, None].astype(np.float32)],
                        axis=1,
                    )
                    center = centers(
                        model,
                        dict(batch, parent=inherited, features=features),
                        neural=neural,
                        device=device,
                    )
                    probability = (
                        gaussian_log_probabilities(center, values, grid.scale).exp().cpu().numpy()
                    )
                    predictions.update(
                        prediction=grid.median(probability), center=center.cpu().numpy()
                    )
                for name, prediction in predictions.items():
                    errors[name].update(
                        batch["market"], batch["prediction_at"], prediction - batch["target"]
                    )
                yield pa.table(
                    dict(
                        sample_id=batch["sample_ids"],
                        asset_id=["/".join(key.split("/")[:2]) for key in batch["sample_ids"]],
                        market=batch["market"],
                        prediction_at=pa.array(
                            batch["prediction_at"], type=pa.timestamp("us", tz="UTC")
                        ),
                        target=batch["target"],
                        **predictions,
                    )
                )

    atomic_parquet_batches(destination, tables())
    scores = {name: value.summary() for name, value in errors.items()}
    count = dataset.manifest["counts"][partition]
    if not count or any(value["samples"] != count for value in scores.values()):
        raise ValueError("La evaluación no concilia todas las filas del bloque")
    for value in scores.values():
        value.update(mae=value["absolute_error"] / count, mse=value["squared_error"] / count)
    return scores


def _jobs(reference, tabular, adjustments, *, arm="US"):
    if arm not in ("US", "CN", "US+CN"):
        raise ValueError("La evaluación temporal requiere un único brazo declarado")
    adjustment_summary, _ = read_manifest(adjustments, 8 * 1024**2)
    seeds = matching_seeds(adjustment_summary["identity"]["proof"])
    proof = (
        matching_parents(reference, tabular, arm, seeds=seeds)
        if seeds is not None
        else selected_parents(reference, tabular, arm)
    )
    summaries, jobs = {}, []
    for stage, path in (
        ("reference", reference),
        ("tabular", tabular),
        ("posttraining", adjustments),
    ):
        report, digest = read_manifest(path, 8 * 1024**2)
        if (
            report.get("status") != "completed"
            or report.get("final_test_opened") is not False
            or report.get("completed_runs") != report.get("planned_runs")
        ):
            raise ValueError("La evaluación necesita todas las ejecuciones confirmadas")
        summaries[stage] = dict(path=str(path.resolve()), sha256=digest)
        runs = report["runs"]
        if len(runs) != report["planned_runs"]:
            raise ValueError("Faltan recibos de ejecuciones en la selección congelada")
        if stage == "posttraining":
            if report["identity"]["proof"] != proof:
                raise ValueError("Los ajustes no corresponden a las referencias congeladas")
            entries = (
                (key, row["path"], row["sha256"], None, "posttraining") for key, row in runs.items()
            )
        else:
            by_id = {row["id"]: row for row in runs}
            entries = (
                (
                    row["id"],
                    (row["path"] if stage == "reference" else row["attempts"][-1]["path"])
                    + "/run.json",
                    row["report_sha256"],
                    row.get("parent"),
                    row["stage"],
                )
                for row in runs
            )
        for name, relative, signature, parent_id, phase in entries:
            path_run, _ = _verified_file(
                path.parent, dict(path=relative, sha256=signature), maximum_bytes=8 * 1024**2
            )
            run, _ = read_manifest(path_run, 8 * 1024**2)
            if run.get("status") != "completed" or run.get("final_test_opened") is not False:
                raise ValueError("Una selección no corresponde a un resultado completo")
            _verified_file(path_run.parent, run["checkpoint"])
            comparator = None
            if parent_id is not None:
                inherited = by_id[parent_id]
                parent_path, _ = _verified_file(
                    path.parent,
                    dict(path=inherited["path"] + "/run.json", sha256=inherited["report_sha256"]),
                    maximum_bytes=8 * 1024**2,
                )
                parent_report, _ = read_manifest(parent_path, 8 * 1024**2)
                if (
                    run["identity"]["initialization"]["parent_checkpoint_sha256"]
                    != parent_report["checkpoint"]["sha256"]
                ):
                    raise ValueError("La continuación no corresponde a los pesos de su padre")
                comparator = dict(
                    path=str(parent_path.resolve()), sha256=inherited["report_sha256"]
                )
            jobs.append(
                dict(
                    id=f"{stage}/{name}",
                    stage=stage,
                    report=str(path_run.resolve()),
                    sha256=signature,
                    comparator=comparator,
                    phase=phase,
                )
            )
    if len(jobs) > 512 or len({row["id"] for row in jobs}) != len(jobs):
        raise ValueError("Las selecciones están duplicadas o exceden el presupuesto")
    return proof, summaries, jobs


def _adjustment(path, report, parent, device):
    identity = report["identity"]
    if identity["parent"] != parent.identity or identity["code"] != code_identity():
        raise ValueError("El ajuste no conserva la identidad del padre o su implementación")
    grid = ActionGrid.from_dict(identity["grid"])
    normalization = identity["normalization"]
    if (
        normalization["fit_partition"] != "train"
        or normalization["parent_sha256"] != parent.identity["checkpoint_sha256"]
    ):
        raise ValueError("La normalización no procede del entrenamiento del padre")
    neural = identity["case"]["mode"].startswith("neural_")
    model = (
        parent.continuation()
        if neural
        else LinearResidualPolicy(
            normalization["mean"],
            normalization["scale"],
            target_scale=grid.scale,
        )
    )
    state = _best_state(
        path.parent, identity, report["selection"], expected_sha256=report["checkpoint"]["sha256"]
    )
    model.load_state_dict(state["model"], strict=True)
    if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
        raise ValueError("El modelo seleccionado contiene pesos no finitos")
    return model.to(device).eval().requires_grad_(False), grid, neural


def _confirmed_results(output, summary, jobs):
    by_id = {job["id"]: job for job in jobs}
    if (
        summary.get("final_test_opened") is not False
        or summary.get("planned_runs") != len(jobs)
        or summary.get("completed_runs") != len(summary["runs"])
        or set(summary["runs"]) - set(by_id)
        or summary["status"] == "completed"
        and len(summary["runs"]) != len(jobs)
    ):
        raise ValueError("Los recibos no concilian con las evaluaciones previstas")
    for key, saved in summary["runs"].items():
        result_path, _ = _verified_file(output, saved, maximum_bytes=8 * 1024**2)
        result, _ = read_manifest(result_path, 8 * 1024**2)
        if (
            result["job"] != by_id[key]
            or result["status"] != "completed"
            or result.get("final_test_opened") is not False
            or set(result["predictions"]) != set(PARTITIONS)
        ):
            raise ValueError("El recibo de evaluación ha cambiado")
        for record in result["predictions"].values():
            _verified_file(result_path.parent, record)


def run_evaluation(reference, tabular, adjustments, output, *, arm="US", stop=None):
    reference, tabular, adjustments, output = map(Path, (reference, tabular, adjustments, output))
    proof, sources, jobs = _jobs(reference, tabular, adjustments, arm=arm)
    ordered = adjustments.parent / "ordered/manifest.json"
    source, ordered_hash = read_manifest(ordered, 8 * 1024**2)
    if source.get("source_manifest", {}).get("sha256") != proof["manifest_sha256"]:
        raise ValueError("Falta el vínculo temporal de los datos ordenados")
    safe_destination(output)
    for protected in (
        reference.parent,
        tabular.parent,
        adjustments.parent,
        Path(proof["manifest"]).parent,
    ):
        outside_source(protected, output)
        outside_source(output, protected)
    identity = dict(
        arm=arm,
        sources=sources,
        jobs=jobs,
        ordered_sha256=ordered_hash,
        manifest_sha256=proof["manifest_sha256"],
        partitions=list(PARTITIONS),
        code=code_identity()
        | {
            f"posttraining/{name}": sha256(Path(__file__).with_name(name))
            for name in ("heldout.py", "parent_selection.py")
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output / ".evaluation.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    stop = stop or StopRequest()
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        confirmed = initialize_receipt(
            output, identity, record="summary.json", lock=".evaluation.lock"
        )
        summary = (
            read_manifest(output / "summary.json", 8 * 1024**2)[0]
            if confirmed
            else dict(
                schema_version=1,
                kind="frozen_temporal_evaluation",
                identity=identity,
                frozen_at_utc=datetime.now(UTC).isoformat(),
                status="running",
                runs={},
                planned_runs=len(jobs),
                completed_runs=0,
                final_test_opened=False,
            )
        )
        if summary["identity"] != identity:
            raise ValueError("La evaluación pertenece a otra selección congelada")
        _confirmed_results(output, summary, jobs)
        if summary["status"] == "completed":
            return summary
        summary["status"] = "running"
        summary.pop("error", None)
        atomic_json(output / "summary.json", summary)
        began = time.perf_counter()
        try:
            with GpuLease() as lease:
                torch.set_num_threads(4)
                torch.use_deterministic_algorithms(True)
                torch.backends.cudnn.benchmark = False
                dataset = CorpusDataset(Path(proof["manifest"]))
                for job in jobs:
                    if stop.requested:
                        raise InterruptedError("Evaluación pausada")
                    if job["id"] in summary["runs"]:
                        continue
                    path = Path(job["report"])
                    report, digest = read_manifest(path, 8 * 1024**2)
                    if digest != job["sha256"]:
                        raise ValueError("Ha cambiado un modelo congelado")
                    model, grid, neural, predictor = None, None, False, None
                    if job["stage"] == "posttraining":
                        family = job["id"].split("/")[1]
                        selected_parent = parent_for_seed(
                            proof, family, report["identity"]["case"]["seed"]
                        )
                        if sha256(Path(selected_parent["report"])) != selected_parent["sha256"]:
                            raise ValueError(
                                "El padre emparejado cambió tras congelar la evaluación"
                            )
                        parent = load_parent(ordered, Path(selected_parent["report"]), lease=lease)
                        model, grid, neural = _adjustment(path, report, parent, "cuda:0")
                    else:
                        parent = load_parent(ordered, path, lease=lease)
                        if job["comparator"] is not None:
                            record = job["comparator"]
                            if sha256(Path(record["path"])) != record["sha256"]:
                                raise ValueError("El padre congelado ha cambiado")
                            predictor = parent
                            parent = load_parent(ordered, Path(record["path"]), lease=lease)
                    folder = output / "runs" / job["id"]
                    folder.mkdir(parents=True, exist_ok=True)
                    started = time.perf_counter()
                    predictions = {}
                    for partition in PARTITIONS:
                        destination = folder / f"{partition}.parquet"
                        metrics = evaluate_partition(
                            dataset,
                            parent,
                            partition,
                            destination,
                            stop=stop,
                            device="cuda:0",
                            model=model,
                            grid=grid,
                            neural=neural,
                            predictor=predictor,
                        )
                        predictions[partition] = dict(
                            path=destination.name, sha256=sha256(destination), metrics=metrics
                        )
                        lease.check()
                    if sha256(path) != job["sha256"]:
                        raise ValueError("El modelo cambió durante la evaluación")
                    result = dict(
                        schema_version=1,
                        status="completed",
                        job=job,
                        family=parent.kind,
                        parent_comparison=model is not None or predictor is not None,
                        case=(
                            report.get("identity", {}).get("case")
                            or report.get("identity", {}).get("options")
                            or {"alpha": report.get("alpha")}
                        ),
                        checkpoint=report["checkpoint"],
                        selection=report.get("selection"),
                        predictions=predictions,
                        elapsed_seconds=time.perf_counter() - started,
                        primary="median" if model is not None else "continuous",
                        final_test_opened=False,
                    )
                    atomic_json(folder / "run.json", result)
                    summary["runs"][job["id"]] = dict(
                        path=str((folder / "run.json").relative_to(output)),
                        sha256=sha256(folder / "run.json"),
                    )
                    summary["completed_runs"] = len(summary["runs"])
                    summary["updated_at_utc"] = datetime.now(UTC).isoformat()
                    atomic_json(output / "summary.json", summary)
                    del parent, model, predictor
                    gc.collect()
                    torch.cuda.empty_cache()
                summary["resources"] = dict(
                    lease.record,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    peak_torch_vram_bytes=torch.cuda.max_memory_allocated(0),
                )
            summary["status"] = "completed"
        except InterruptedError:
            summary["status"] = "paused"
        except BaseException as error:
            summary.update(
                status="failed", error=dict(type=type(error).__name__, message=str(error))
            )
            raise
        finally:
            summary["last_attempt_seconds"] = time.perf_counter() - began
            summary["updated_at_utc"] = datetime.now(UTC).isoformat()
            atomic_json(output / "summary.json", summary)
        return summary
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reference", "tabular", "adjustments", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=("US", "CN", "US+CN"), default="US")
    args = vars(parser.parse_args())
    with StopRequest() as stop:
        result = run_evaluation(**args, stop=stop)
    print(
        f"Estado: {result['status']}. "
        f"Evaluaciones: {result['completed_runs']}/{result['planned_runs']}"
    )
    return 0 if result["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
