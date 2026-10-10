"""Medir el coste de las proyecciones de la sección 4.4 con entradas reales, sin optimizador.

`materialize` lee en CPU, antes de reservar la GPU, T instantes de decisión de B activos de
EE. UU. con sesiones comunes dentro del tramo de ajuste de una vista de la campaña. Usa las
mismas funciones que el lector por bloques (`_sample_group`, `_observation`, `_new_batch` y
`_fill_batch`) y guarda las entradas en un archivo compacto. No lee objetivos ni etiquetas.

`measure` carga ese archivo y recorre con la receta de la campaña, para `linear_v1` y
`titans_mac_paper_projections_v2`, un tramo diferenciable de T instantes: forward con grafo,
pinball frente a ceros y backward. Ningún optimizador se construye y los gradientes se
descartan en cada repetición. Mide también la inferencia sin grafo del mismo tramo.
"""

import argparse
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.memory.financial_observations import _observation
from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans.config import PAPER_PROJECTIONS
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import (
    DecisionBatch,
    FinancialInputSpec,
    validated_cpu_batch,
)
from mars_titan.training.corpus_inputs import (
    CorpusDataset,
    _fill_batch,
    _historical_times,
    _new_batch,
)
from mars_titan.training.titans_walk_forward import unfused_attention, window_phases

IDENTITIES = ("linear_v1", PAPER_PROJECTIONS)
VARIANTS = ("mac_online", "mac_frozen")
RECIPE = Path("configs/titans/chronological-training-historical-masked.json")


def _view_fold(dataset):
    """Ventana walk-forward de la vista, que todos sus mercados deben compartir."""
    from mars_titan.training.temporal_contract import temporal_contracts

    folds = [
        c["fold"]
        for c in temporal_contracts(dataset.manifest, input_policy=HISTORICAL_MASKED).values()
    ]
    if not folds or any(fold != folds[0] for fold in folds):
        raise ValueError("La vista no declara una única ventana walk-forward")
    return folds[0]


def _stamps(dataset, asset):
    """Instantes de decisión del archivo de muestras, leyendo solo esa columna."""
    path = dataset._file(asset, "samples")
    return _historical_times(pq.read_table(path, columns=["prediction_at"]))


def _location(path, row):
    """Grupo Parquet y fila local de una fila global del archivo de muestras."""
    with pq.ParquetFile(path) as file:
        start = 0
        for group in range(file.num_row_groups):
            size = file.metadata.row_group(group).num_rows
            if row < start + size:
                return group, row - start
            start += size
    raise ValueError("La fila no pertenece al archivo")


def materialize(view, output, *, flows, instants, warmup_months):
    """Elegir sesiones comunes en mitad del tramo de ajuste y copiar sus entradas.

    Las fechas salen del primer activo de EE. UU. con suficientes sesiones a partir de la
    mitad del tramo, y se toman los primeros activos que las tienen todas. Se exige que
    todas las filas compartan instante porque el entrenador avanza por instantes.
    """
    dataset = CorpusDataset(Path(view), input_policy=HISTORICAL_MASKED)
    phase = window_phases(_view_fold(dataset), warmup_months)["train"]
    middle = phase.decision_start + (phase.decision_end - phase.decision_start) // 2
    candidates = [asset for asset in dataset.assets if asset["market"] == "US"]
    dates, chosen = None, []
    for asset in candidates:
        stamps = _stamps(dataset, asset)
        if dates is None:
            later = stamps[(stamps >= middle) & (stamps < phase.decision_end)]
            if len(later) >= instants:
                dates = later[:instants]
            else:
                continue
        rows = np.searchsorted(stamps, dates)
        if (rows < len(stamps)).all() and (
            stamps[np.minimum(rows, len(stamps) - 1)] == dates
        ).all():
            chosen.append((asset, rows))
        if len(chosen) == flows:
            break
    if len(chosen) != flows:
        raise ValueError("No hay suficientes activos con las sesiones comunes elegidas")
    batches = [None] * instants
    for filled, (asset, rows) in enumerate(chosen):
        path = dataset._file(asset, "samples")
        prices, decoded_groups = dataset._prices(asset), {}
        for index, (row, at) in enumerate(zip(rows.tolist(), dates.tolist(), strict=True)):
            group, local = _location(path, row)
            if group not in decoded_groups:
                with pq.ParquetFile(path) as file:
                    _, *decoded_groups[group] = dataset._sample_group(asset, file, group)
            decoded = decoded_groups[group]
            block = _observation(dataset, asset, decoded, local, at, prices)
            if batches[index] is None:
                batches[index] = _new_batch(
                    decoded[2], dataset.context, flows, masked=True, supervised=False
                )
            _fill_batch(batches[index], filled, block, 0, 1)
    arrays = {}
    for index, batch in enumerate(batches):
        for name, value in batch["inputs"].items():
            arrays[f"{index}/inputs/{name}"] = value
        for name in ("presence", "prediction_at", "input_available_at"):
            arrays[f"{index}/{name}"] = batch[name]
    output = Path(output)
    np.savez(output, **arrays)
    widths = {name: value.shape[-1] for name, value in batches[0]["inputs"].items()}
    metadata = dict(
        schema_version=1,
        view_manifest_sha256=sha256(Path(view)),
        source_sha256=dataset.identity,
        representation=dataset.manifest["representation"],
        input_policy=HISTORICAL_MASKED,
        dimensions=widths,
        phase=dict(
            partition=phase.partition,
            decision_start=phase.decision_start,
            decision_end=phase.decision_end,
        ),
        dates_us=dates.tolist(),
        flows=[f"US/{asset['symbol']}" for asset, _ in chosen],
        sample_ids=[batch["sample_ids"] for batch in batches],
        market=[batch["market"] for batch in batches],
        files_sha256={
            f"US/{asset['symbol']}": dict(
                samples=sha256(dataset._file(asset, "samples")),
                prices=sha256(dataset._file(asset, "prices")),
            )
            for asset, _ in chosen
        },
        data_sha256=sha256(output),
        targets_read=False,
    )
    atomic_json(output.with_suffix(".json"), metadata)
    return metadata


def _raw_batches(data):
    metadata = json.loads(Path(data).with_suffix(".json").read_text())
    if sha256(Path(data)) != metadata["data_sha256"]:
        raise ValueError("El archivo materializado cambió desde su lectura")
    arrays = np.load(data)
    batches = []
    for index in range(len(metadata["dates_us"])):
        names = metadata["dimensions"]
        batches.append(
            dict(
                inputs={name: arrays[f"{index}/inputs/{name}"] for name in names},
                presence=arrays[f"{index}/presence"],
                sample_ids=metadata["sample_ids"][index],
                market=metadata["market"][index],
                prediction_at=arrays[f"{index}/prediction_at"],
                input_available_at=arrays[f"{index}/input_available_at"],
            )
        )
    spec = FinancialInputSpec(
        source_sha256=metadata["source_sha256"],
        view_sha256=metadata["view_manifest_sha256"],
        representation=metadata["representation"],
        dimensions=metadata["dimensions"],
        input_policy=metadata["input_policy"],
    )
    return metadata, spec, batches


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _summary(values):
    ordered = sorted(values)
    return dict(
        median=statistics.median(ordered),
        p95=ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        minimum=ordered[0],
        maximum=ordered[-1],
    )


def _case(spec, batches, options, projections, variant, device, repetitions, warmup):
    """Repetir el tramo con una identidad y una variante y resumir tiempos y memoria.

    El estado inicial se crea fuera del cronómetro porque comprueba la huella de los
    parámetros. Los gradientes se descartan antes de cada repetición para que el backward
    no acumule sobre los anteriores, ya que no hay optimizador que los consuma.
    """
    model = FinancialPredictor(
        FinancialConfig(spec, variant=variant, seed=42, memory_projections=projections, **options),
        device=device,
    )
    model.train()
    flows = batches[0].flow_ids
    parameters = list(model.parameters())
    records = []
    for repetition in range(warmup + repetitions):
        for parameter in parameters:
            parameter.grad = None
        state = model.initial_state(flows, differentiable=True)
        _synchronize(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        outputs = []
        for batch in batches:
            prepared = model.prepare(batch, state, differentiable=True)
            state = prepared.next_state
            outputs.append(prepared.quantiles)
        _synchronize(device)
        forward = time.perf_counter() - started
        prediction = torch.cat(outputs)
        loss = pinball_loss(prediction, torch.zeros(len(prediction), device=device))
        started = time.perf_counter()
        loss.backward()
        _synchronize(device)
        backward = time.perf_counter() - started
        record = dict(forward_with_graph_seconds=forward, backward_seconds=backward)
        if device.type == "cuda":
            record.update(
                allocated_before_bytes=before,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
            )
        del prediction, loss, outputs, prepared, state
        state = model.initial_state(flows)
        _synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            for batch in batches:
                state = model.prepare(batch, state).next_state
        _synchronize(device)
        record["inference_seconds"] = time.perf_counter() - started
        if repetition >= warmup:
            records.append(record)
    for parameter in parameters:
        parameter.grad = None
    rows = len(flows) * len(batches)
    result = {name: _summary([record[name] for record in records]) for name in records[0]}
    step = [r["forward_with_graph_seconds"] + r["backward_seconds"] for r in records]
    result["rows_per_second_forward_backward"] = rows / statistics.median(step)
    result["rows_per_second_inference"] = rows / statistics.median(
        [r["inference_seconds"] for r in records]
    )
    result["parameters"] = sum(p.numel() for p in parameters)
    result["tensor_bytes_per_flow"] = model._tensor_bytes_per_flow()
    result["repetitions"] = records
    return result


def _nvidia_smi():
    try:
        query = "name,driver_version,memory.total,memory.used,utilization.gpu,pstate"
        return subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        return f"no disponible: {error}"


def measure(data, output, *, device, repetitions, warmup, threads):
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Se pidió cuda:0 y no hay dispositivo CUDA visible")
        torch.cuda.init()
    started = time.perf_counter()
    metadata, spec, raws = _raw_batches(data)
    document = json.loads(RECIPE.read_text())
    options = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    if document["predictor"]["dtype"] != "float32":
        raise ValueError("La medida reproduce la receta FP32")
    batches = [
        DecisionBatch.from_validated(
            validated_cpu_batch(raw, spec), device=device, dtype=torch.float32
        )
        for raw in raws
    ]
    smi_before = _nvidia_smi() if device.type == "cuda" else None
    free = torch.cuda.mem_get_info(device) if device.type == "cuda" else None
    cases = {}
    with unfused_attention():
        for variant in VARIANTS:
            for projections in IDENTITIES:
                cases[f"{variant}/{projections}"] = _case(
                    spec, batches, options, projections, variant, device, repetitions, warmup
                )
                print(variant, projections, flush=True)
    ratios = {}
    for variant in VARIANTS:
        old, new = cases[f"{variant}/linear_v1"], cases[f"{variant}/{PAPER_PROJECTIONS}"]
        ratios[variant] = {
            name: new[name]["median"] / old[name]["median"]
            for name in ("forward_with_graph_seconds", "backward_seconds", "inference_seconds")
        }
        if device.type == "cuda":
            ratios[variant]["peak_allocated_bytes"] = (
                new["peak_allocated_bytes"]["maximum"] / old["peak_allocated_bytes"]["maximum"]
            )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    receipt = dict(
        schema_version=1,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=commit,
        command=(
            "python benchmarks/titans_paper_projections_cost.py measure --data <entradas.npz> "
            f"--device {device} --repetitions {repetitions} --warmup {warmup} --output <recibo>"
        ),
        scope=(
            "Tramo diferenciable de la receta de la campaña con entradas reales leídas antes "
            "de reservar la GPU. Pinball frente a ceros solo para obtener un backward. No es "
            "un ajuste ni una evaluación."
        ),
        data=dict(
            path="<entradas.npz>",
            sha256=metadata["data_sha256"],
            view_manifest_sha256=metadata["view_manifest_sha256"],
            source_sha256=metadata["source_sha256"],
            flows=len(metadata["flows"]),
            instants=len(metadata["dates_us"]),
            dates_us=metadata["dates_us"],
            dimensions=metadata["dimensions"],
            targets_read=metadata["targets_read"],
        ),
        recipe=str(RECIPE),
        recipe_sha256=sha256(RECIPE),
        predictor=document["predictor"],
        hardware=dict(
            machine=platform.machine(),
            device=str(device),
            cuda_device=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            capability=list(torch.cuda.get_device_capability(device))
            if device.type == "cuda"
            else None,
            nvidia_smi_before=smi_before,
            free_and_total_bytes_before=list(free) if free else None,
            nvidia_smi_after=_nvidia_smi() if device.type == "cuda" else None,
        ),
        versions=dict(
            python=platform.python_version(),
            torch=torch.__version__,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version() if device.type == "cuda" else None,
            numpy=np.__version__,
        ),
        numerics=dict(
            dtype="float32",
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
            tf32_cudnn=torch.backends.cudnn.allow_tf32,
            autocast=False,
            mha_fastpath=False,
        ),
        threads=threads,
        warmup=warmup,
        repetitions=repetitions,
        cases=cases,
        paper_over_linear_median_ratio=ratios,
        optimizer_steps=0,
        optimizers_constructed=0,
        training_runs=0,
        gpu_workloads=int(device.type == "cuda"),
        seconds=time.perf_counter() - started,
    )
    atomic_json(Path(output), receipt)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("materialize")
    read.add_argument("--view", type=Path, required=True)
    read.add_argument("--output", type=Path, required=True)
    read.add_argument("--flows", type=int, default=128)
    read.add_argument("--instants", type=int, default=8)
    read.add_argument("--warmup-months", type=int, default=12)
    run = commands.add_parser("measure")
    run.add_argument("--data", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    run.add_argument("--repetitions", type=int, default=20)
    run.add_argument("--warmup", type=int, default=3)
    run.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.command == "materialize":
        metadata = materialize(
            args.view,
            args.output,
            flows=args.flows,
            instants=args.instants,
            warmup_months=args.warmup_months,
        )
        print(json.dumps({key: metadata[key] for key in ("dates_us", "dimensions")}))
    else:
        measure(
            args.data,
            args.output,
            device=args.device,
            repetitions=args.repetitions,
            warmup=args.warmup,
            threads=args.threads,
        )


if __name__ == "__main__":
    main()
