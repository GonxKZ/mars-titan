"""Regenerar por inferencia las predicciones por fila de un trabajo confirmado.

La retención v2 libera las tablas por fila cuando sus agregados ya están guardados. Esta
orden es la que permite conservarlas de forma reproducible: vuelve a predecir, desde el
estado elegido que se conserva siempre, las mismas filas con el mismo código y lote, y
compara el resultado bit a bit con la huella de contenido registrada.

- Un ajuste se regenera con el traslado de su familia sobre su propia ventana y con su
  intento como ancla (`regenerate=True`): validación, calibración y evaluación.
- Un traslado se regenera repitiendo el mismo traslado desde el mismo ancla.
- Una predicción de la ablación de modalidades se regenera con el mismo ejecutor de la
  etapa, la misma variante y el mismo estado de partida.

El proceso fija FP32 estricto, sin TF32 en cuBLAS ni en cuDNN. Los traslados comprueban
además la identidad numérica que registró el ajuste. Si una tabla no sale idéntica, el
informe lo dice y la retención la conserva. La regeneración no ajusta pesos, selección ni
normalizadores, y escribe en un destino nuevo, nunca sobre las tablas originales. Como
predice ventanas reales de la campaña, se detiene con el bloqueo de aprendizaje igual que
los traslados y la ablación.
"""

import argparse
import dataclasses
import json
from functools import partial
from pathlib import Path

import pyarrow.parquet as pq

from mars_titan.data import prediction_files
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import outside_source, sha256

KIND = "prediction_regeneration"


class _Running:
    requested = False


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_fp32():
    """FP32 estricto en el proceso: sin TF32 en cuBLAS ni en cuDNN."""
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return dict(
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )


_TF32 = ("cudnn_allow_tf32", "cuda_matmul_allow_tf32", "matmul_allow_tf32")
_PRECISION = ("float32_matmul_precision", "matmul_precision")


def recorded_numerics(document):
    """Políticas numéricas que registra un informe, a cualquier profundidad."""
    found = []
    if isinstance(document, dict):
        if "cudnn_allow_tf32" in document:
            found.append({k: document[k] for k in _TF32 + _PRECISION if k in document})
        for value in document.values():
            found += recorded_numerics(value)
    elif isinstance(document, list):
        for value in document:
            found += recorded_numerics(value)
    return found


def require_strict(report, label):
    """Rechazar la regeneración de un trabajo que no se ejecutó en FP32 estricto.

    Regenerar en FP32 estricto lo que se predijo con TF32 no repetiría la misma precisión,
    así que esas filas se conservan. Los modelos tabulares no registran política de PyTorch.
    """
    for numerics in recorded_numerics(report):
        _require(
            not any(numerics.get(key) for key in _TF32)
            and all(numerics[key] == "highest" for key in _PRECISION if key in numerics),
            f"{label} no se predijo en FP32 estricto: sus filas se conservan",
        )


def originals(folder, report):
    """Tablas por fila de un informe de ajuste o traslado, con su ruta y su huella."""
    return {
        partition: (Path(folder) / record["path"], record["sha256"])
        for partition, record in report["predictions"].items()
        if partition != "train"
    }


def compare(expected, produced_folder, report):
    """Comparar cada tabla regenerada con la registrada, sin aceptar diferencias."""
    _require(
        set(expected) <= set(report["predictions"]),
        "La regeneración no repite todos los tramos registrados",
    )
    partitions = {}
    for partition, (path, digest) in expected.items():
        regenerated = Path(produced_folder) / report["predictions"][partition]["path"]
        table = pq.read_table(regenerated, use_threads=False)
        partitions[partition] = dict(
            original=str(path),
            regenerated=str(regenerated),
            state=prediction_files.verify(path, digest),
            rows=table.num_rows,
            identical=prediction_files.matches(path, digest, table),
            same_file=sha256(regenerated) == digest,
        )
    return dict(
        identical=all(item["identical"] for item in partitions.values()),
        partitions=partitions,
    )


def _destination(destination, *protected):
    destination = Path(destination)
    safe_destination(destination)
    _require(not destination.exists(), "La regeneración necesita un destino nuevo")
    for source in protected:
        outside_source(source, destination)
    return destination


def _confirmed_until(state, jobs, job_id):
    """Confirmar en orden los recibos de la campaña hasta el trabajo pedido."""
    for job in jobs:
        case, _, sources = state.resolve(job)
        receipt = state.confirmed(job, state.job_identity(job, case, sources))
        _require(receipt is not None, f"Falta confirmar {job['id']}")
        state.receipts[job["id"]] = receipt
        if job["id"] == job_id:
            return job, case, receipt
    raise ValueError(f"{job_id} no es un trabajo de la campaña")


def regenerate_job(path, views, output, job_id, destination, *, regenerators=None):
    """Regenerar las predicciones de un trabajo confirmado de la campaña base.

    `regenerators` sustituye los regeneradores por modelo y tipo, como en las pruebas.
    Devuelve el informe de comparación, que también se escribe en el destino.
    """
    from . import masked_campaign as engine
    from .campaign_plan import plan_campaign
    from .learning_hold import require_learning_allowed

    require_learning_allowed("la regeneración de predicciones de la campaña")
    campaign, state = engine._confirmed_state(path, views, output)
    destination = _destination(destination, Path(output))
    job, _, _ = _confirmed_until(state, plan_campaign(campaign), job_id)
    return _regenerate_base(campaign, state, job, destination, regenerators)


def _regenerate_base(campaign, state, job, destination, regenerators=None):
    """Regenerar un trabajo cuyos recibos y dependencias ya están confirmados en `state`.

    La meseta de un ajuste conjunto no tiene tablas que regenerar: el estado elegido y las
    predicciones son los de su continuación, así que se rechaza antes de leer nada.
    """
    from . import masked_campaign as engine
    from .campaign_plan import FIT, PLATEAU

    _require(
        job.get("phase") != PLATEAU,
        f"{job['id']} es una meseta sin predicciones: se regenera su continuación",
    )
    receipt = state.receipts[job["id"]]
    case = state.resolve(job)[0]
    report_path = state.output / receipt["report"]["path"]
    report, _ = read_manifest(report_path, 16 * 1024**2)
    require_strict(report, job["id"])
    expected = originals(report_path.parent, report)
    view = state.views[job["scope"]]["windows"][job["window"]]
    section = campaign["neural" if job["family"] == engine.NEURAL else "tabular"]
    if job["kind"] == FIT:
        attempt = state.output / receipt["attempt"]
        anchor = dict(folder=attempt, view=Path(view["path"]), job=job["id"])
    else:
        anchor = state.resolve(job)[1]
    run = engine.JobRun(
        job=job,
        case=case,
        view=Path(view["path"]),
        view_sha256=view["sha256"],
        folder=Path(destination) / "run",
        policy=campaign["input_policy"],
        batch_size=section["batch_size"],
        checkpoint_seconds=campaign["neural"]["checkpoint_seconds"],
        stop=_Running(),
        anchor=anchor,
        parent=state.parent_of(job),
    )
    available = engine.regenerators() if regenerators is None else regenerators
    regenerate = available.get((job["model"], job["kind"]))
    _require(regenerate is not None, f"{job['model']} no tiene regeneración: se conserva")
    numerics = strict_fp32()
    Path(destination).mkdir(parents=True)
    produced = regenerate(run)
    return _publish(
        destination,
        dict(job=job["id"], stage="base", model=job["model"], job_kind=job["kind"]),
        numerics,
        compare(expected, run.folder, produced),
    )


def regenerate_ablation(path, views, campaign_output, output, job_id, destination, **options):
    """Regenerar la evaluación enmascarada de un trabajo confirmado de la ablación."""
    from . import modality_ablation_stage as ablation
    from .learning_hold import require_learning_allowed

    require_learning_allowed("la regeneración de predicciones de la ablación")
    stage, base, output = ablation._opened(path, views, campaign_output, output)
    destination = _destination(destination, Path(campaign_output), output)
    identity = ablation._identity(stage, base.views)
    executors = options.get("executors") or ablation.EXECUTORS
    state = ablation._Stage(stage, base, output, identity, executors, _Running())
    jobs = {job["id"]: job for job in ablation.plan_stage(stage)}
    _require(job_id in jobs, f"{job_id} no es un trabajo de la etapa")
    job = jobs[job_id]
    receipt = state.confirmed(job, state.job_identity(job))
    _require(receipt is not None, f"Falta confirmar {job_id}")
    state.receipts[job_id] = receipt
    return _regenerate_ablation(state, job, destination)


def _regenerate_ablation(state, job, destination):
    """Regenerar una predicción enmascarada con el recibo ya confirmado en `state`."""
    from . import modality_ablation_stage as ablation

    receipt = state.receipts[job["id"]]
    require_strict(
        read_manifest(state.output / receipt["report"]["path"], 16 * 1024**2)[0], job["id"]
    )
    record = receipt["prediction"]
    expected = {ablation.PARTITION: (state.output / record["path"], record["sha256"])}
    run = dataclasses.replace(
        state.prepare(job, state.job_identity(job)), folder=Path(destination) / "run"
    )
    numerics = strict_fp32()
    Path(destination).mkdir(parents=True)
    produced = state.executors[job["model"]](run)
    return _publish(
        destination,
        dict(job=job["id"], stage="ablation", model=job["model"], variant=job["variant"]),
        numerics,
        compare(expected, run.folder, produced),
    )


def _publish(destination, job, numerics, comparison):
    from mars_titan.data.storage import atomic_json

    report = dict(
        schema_version=1,
        kind=KIND,
        **job,
        numerics=numerics,
        **comparison,
        final_test_opened=False,
        optimizer_steps=0,
    )
    atomic_json(destination / "regeneration.json", report)
    return report


def regenerator(executor, **options):
    """Regenerador a partir del ejecutor de traslado de una familia."""
    return partial(executor, regenerate=True, **options)


def main(argv=None):
    from .masked_campaign import _views_argument

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--views", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True, help="Salida de la campaña base")
    parser.add_argument("--job", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--ablation-stage", type=Path, help="Etapa de ablación del trabajo")
    parser.add_argument("--ablation-output", type=Path)
    args = parser.parse_args(argv)
    views = _views_argument(args.views)
    if args.ablation_stage is None:
        report = regenerate_job(args.campaign, views, args.output, args.job, args.destination)
    else:
        _require(args.ablation_output is not None, "Falta la salida de la etapa de ablación")
        report = regenerate_ablation(
            args.ablation_stage,
            views,
            args.output,
            args.ablation_output,
            args.job,
            args.destination,
        )
    print(json.dumps(dict(job=report["job"], identical=report["identical"]), ensure_ascii=False))
    return 0 if report["identical"] else 1
