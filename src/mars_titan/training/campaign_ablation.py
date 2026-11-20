"""Ablación de modalidades de una campaña comparativa, sin reentrenar ni tocar sus trabajos.

Para cada trabajo terminado de los modelos elegidos (ajustes de Titans-MAC y lectores de
MARS-TITAN) y cada variante de `data.modality_ablation` (`mask_news`, `mask_fundamentals` y
`mask_news_and_fundamentals`), vuelve a predecir la evaluación de su propia ventana con esas
modalidades leídas como una ausencia real: bit de presencia falso, relleno de ausencia,
disponibilidad nula, cero noticias admitidas y la causa `modality_ablation`, también en el
calentamiento, así que la memoria rápida y el banco episódico ven la misma ausencia. Reutiliza
el traslado de cada familia con `modality_ablation` (`carry_titans` y `carry_mars_titan_arm`):
no ajusta pesos ni selección. Precios, gráficos y macro no se ablacionan porque nunca faltan en
la edición con máscaras; retirarlos mediría entradas fuera de distribución.

Cada ablación vive en `<estado>/ablations/<variante>/<modelo>/<origen>/seed-<semilla>/` con la
evaluación y su recibo `carry.json`. Una ablación con recibo se comprueba (variante, tramo,
huella del informe del trabajo y de sus predicciones) y no se repite; un recibo incoherente
detiene la orden. Una carpeta sin recibo es un intento interrumpido sin nada confirmado y se
descarta antes de repetirlo; un cerrojo exclusivo impide que dos órdenes recorran el mismo
estado a la vez, porque una descartaría la carpeta que la otra está escribiendo. Cada tarea
corre en un proceso nuevo con el entorno de los
trabajos de la campaña y, con `cuda_gib`, su mismo tope de memoria en la GPU. El resumen de la
orden queda en `<estado>/ablations/summary.json`.
"""

import argparse
import fcntl
import json
import multiprocessing
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.modality_ablation import VARIANTS, ablation_identity
from mars_titan.data.storage import atomic_json, sha256

from . import comparison_campaign as campaign

KIND = "comparison_campaign_modality_ablation_v1"
FOLDER = "ablations"
PARTITION = "evaluation"
RECEIPT = "carry.json"
# Solo los trabajos con estado elegido por semilla tienen un traslado con ablación.
RUNNERS = ("fit", "reader")
MAX_WORKERS = 8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def destination(layout, variant, job):
    """Carpeta de la ablación de un trabajo con una variante."""
    return layout.state / FOLDER / variant / job.key


def selected(declaration, machine, models, variants):
    """Trabajos de los modelos elegidos en el orden del plan y las variantes comprobadas."""
    models, variants = list(models), list(variants)
    _require(
        models and len(set(models)) == len(models) and set(models) <= set(declaration["models"]),
        "Elige modelos declarados en la campaña, sin repetir",
    )
    _require(
        all(declaration["models"][model]["runner"] in RUNNERS for model in models),
        "Solo los ajustes de Titans-MAC y los lectores de MARS-TITAN tienen ablación",
    )
    _require(
        variants and len(set(variants)) == len(variants) and set(variants) <= set(VARIANTS),
        f"Las variantes son las de data.modality_ablation: {', '.join(VARIANTS)}",
    )
    jobs = [job for job in campaign.plan(declaration, machine) if job.model in models]
    return jobs, variants


def checked_receipt(folder, anchor, variant):
    """Recibo comprobado de la ablación en `folder` del trabajo cuyo informe es `anchor`.

    Devuelve None si todavía no hay recibo. Un recibo de otra variante, de otro tramo, de otro
    informe del trabajo o con predicciones cambiadas detiene la lectura.
    """
    folder = Path(folder)
    path = folder / RECEIPT
    if not path.is_file():
        return None
    receipt, _ = read_manifest(path, 4 * 1024**2)
    predictions = receipt.get("predictions")
    origin = receipt.get("anchor")
    _require(
        receipt.get("status") == "completed"
        and receipt.get("final_test_opened") is False
        and receipt.get("scientific_training_started") is False
        and receipt.get("modality_ablation") == ablation_identity(variant)
        and isinstance(predictions, dict)
        and list(predictions) == [PARTITION]
        and isinstance(origin, dict)
        and origin.get("run_sha256") == sha256(Path(anchor)),
        f"La ablación de {folder} no corresponde a su trabajo, su variante o su tramo",
    )
    entry = predictions[PARTITION]
    _require(
        sha256(folder / entry["path"]) == entry["sha256"],
        f"Han cambiado las predicciones de {folder}",
    )
    return receipt


def verified(layout, job, variant):
    """Recibo comprobado de la ablación de un trabajo, o None si todavía no hay recibo."""
    return checked_receipt(
        destination(layout, variant, job), layout.output(job.key) / "run.json", variant
    )


def discard_incomplete(layout, variant, job):
    """Descartar la carpeta de un intento interrumpido y decir si había alguna."""
    folder = destination(layout, variant, job)
    if not folder.exists() and not folder.is_symlink():
        return False
    safe_destination(folder)
    _require(
        folder.is_dir()
        and not folder.is_symlink()
        and folder.resolve().is_relative_to((layout.state / FOLDER).resolve())
        and not (folder / RECEIPT).exists(),
        f"{folder} no es un intento interrumpido de esta ablación",
    )
    shutil.rmtree(folder)
    return True


def run_task(state, views, declaration, machine, key, variant, cuda_gib, device):
    """Una ablación en este proceso, con el entorno de los trabajos de la campaña."""
    layout = campaign.Layout(state, views)
    job = next(job for job in campaign.plan(declaration, machine) if job.key == key)
    os.environ.update(campaign.job_environment(layout, declaration, job))
    from .checkpoints import StopRequest
    from .financial_run import Paused
    from .mars_titan_walk_forward import carry_mars_titan_arm
    from .titans_walk_forward import carry_titans, unfused_attention

    if cuda_gib is not None and device.startswith("cuda"):
        from . import campaign_slots

        campaign_slots.bound_vram(int(cuda_gib * 2**30))
    carry = carry_titans if job.runner == "fit" else carry_mars_titan_arm
    view = layout.cohort(job) / "manifest.json"
    folder = destination(layout, variant, job)
    folder.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    try:
        with StopRequest() as stop, unfused_attention():
            receipt = carry(
                layout.output(job.key),
                view,
                view,
                folder,
                device=device,
                stop=stop,
                modality_ablation=variant,
            )
    except Paused:
        return dict(status="paused", seconds=time.perf_counter() - started)
    return dict(status=receipt["status"], seconds=time.perf_counter() - started)


def _pool(workers):
    # Un proceso nuevo por tarea: entorno, contexto CUDA y tope de memoria propios.
    context = multiprocessing.get_context("spawn")
    return ProcessPoolExecutor(workers, mp_context=context, max_tasks_per_child=1)


def run(
    declaration, layout, machine, models, variants, *, workers=1, cuda_gib=None, device="cuda:0"
):
    """Ejecutar las ablaciones pendientes de los trabajos terminados y devolver el resumen."""
    _require(type(workers) is int and 1 <= workers <= MAX_WORKERS, "Entre 1 y 8 trabajadores")
    _require(
        cuda_gib is None
        or (type(cuda_gib) in (int, float) and 0 < cuda_gib <= campaign.MAX_CUDA_GIB),
        "El tope de la GPU por tarea es positivo y no supera el de la campaña",
    )
    _require(device in ("cpu", "cuda:0"), "El dispositivo debe ser cpu o cuda:0 explícitos")
    jobs, variants = selected(declaration, machine, models, variants)
    root = layout.state / FOLDER
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Otra orden de ablación recorre ya este estado") from error
        return _run(declaration, layout, machine, jobs, variants, models, workers, cuda_gib, device)


def _run(declaration, layout, machine, jobs, variants, models, workers, cuda_gib, device):
    summary = dict(
        schema_version=1,
        kind=KIND,
        started_at_utc=datetime.now(UTC).isoformat(),
        models=list(models),
        variants=variants,
        workers=workers,
        cuda_gib=cuda_gib,
        code_commit=campaign.code_commit(),
        verified=[],
        waiting=[],
        discarded=[],
        completed=[],
        paused=[],
        failed=[],
        final_test_opened=False,
    )
    tasks = []
    for job in jobs:
        if campaign.read_record(layout, job.key)["status"] != campaign.DONE:
            summary["waiting"].append(job.key)
            continue
        for variant in variants:
            name = f"{variant}/{job.key}"
            if verified(layout, job, variant) is not None:
                summary["verified"].append(name)
                continue
            if discard_incomplete(layout, variant, job):
                summary["discarded"].append(name)
            tasks.append((job.key, variant))
    path = layout.state / FOLDER / "summary.json"
    atomic_json(path, summary)
    with _pool(workers) as pool:
        futures = {
            pool.submit(
                run_task,
                str(layout.state),
                str(layout.views),
                declaration,
                machine,
                key,
                variant,
                cuda_gib,
                device,
            ): (key, variant)
            for key, variant in tasks
        }
        for future in as_completed(futures):
            key, variant = futures[future]
            name = f"{variant}/{key}"
            try:
                result = future.result()
            except Exception as error:  # Se registra y la orden termina con error.
                summary["failed"].append(
                    dict(task=name, error_type=type(error).__name__, error=str(error))
                )
            else:
                if result["status"] == "completed":
                    job = next(job for job in jobs if job.key == key)
                    verified(layout, job, variant)
                    summary["completed"].append(dict(task=name, seconds=result["seconds"]))
                else:
                    summary["paused"].append(dict(task=name, seconds=result["seconds"]))
            atomic_json(path, summary)
    summary["finished_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json(path, summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--machine", choices=campaign.MACHINES, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--cuda-gib", type=float)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    declaration, digest = campaign.load_declaration(args.declaration)
    layout = campaign.Layout(args.state.resolve(), args.views.resolve())
    # La ablación corre con el código que la contiene, que puede ser posterior al de la
    # campaña: solo exige el permiso vigente de esta misma declaración. Los traslados no
    # escriben en los trabajos y comprueban la huella de cada informe que leen.
    permission = (
        json.loads(layout.authorization.read_text(encoding="utf-8"))
        if layout.authorization.is_file()
        else {}
    )
    _require(
        permission.get("training_allowed") is True
        and permission.get("declaration_sha256") == digest,
        "La declaración no es la autorizada en este estado de la campaña",
    )
    summary = run(
        declaration,
        layout,
        args.machine,
        args.models,
        args.variants,
        workers=args.workers,
        cuda_gib=args.cuda_gib,
        device=args.device,
    )
    counts = {
        name: len(summary[name])
        for name in ("verified", "waiting", "discarded", "completed", "paused", "failed")
    }
    print(json.dumps(counts, ensure_ascii=False))
    for failure in summary["failed"]:
        print(f"{failure['task']}: {failure['error_type']}: {failure['error']}", file=sys.stderr)
    return 1 if summary["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
