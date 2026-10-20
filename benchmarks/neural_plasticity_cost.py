"""Medir las variantes neuronales en CUDA con entradas reales y sin optimizador."""

import argparse
import json
import math
import os
import platform
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from benchmarks.titans_paper_projections_cost import _raw_batches
from benchmarks.titans_variant_cost import _initial_context, _step
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch
from mars_titan.training.refinement_inference import _model_digest
from mars_titan.training.titans_walk_forward import unfused_attention

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs/experiments/neural-plasticity-v1.json"


def load_inputs(data):
    """Comprobar fechas y alcance antes de abrir los arrays materializados."""
    metadata = json.loads(Path(data).with_suffix(".json").read_text())
    dates = metadata.get("dates_us")
    if (
        metadata.get("targets_read") is not False
        or not isinstance(dates, list)
        or not dates
        or any(type(at) is not int or not 0 <= at < 1704067200000000 for at in dates)
    ):
        raise ValueError("La medida necesita fechas anteriores a 2024 y ausencia de objetivos")
    observed, specification, raws = _raw_batches(data)
    if observed != metadata:
        raise ValueError("Los metadatos cambiaron durante la lectura")
    if any(np.any(raw["prediction_at"] >= np.datetime64("2024-01-01")) for raw in raws):
        raise ValueError("Los arrays contienen entradas del periodo final reservado")
    return metadata, specification, raws


def summary(samples, *, rows):
    """Percentiles lineales y caudal sobre el tiempo total, con unidades explícitas."""
    if (
        not samples
        or any(not math.isfinite(x) or x <= 0 for x in samples)
        or type(rows) is not int
        or rows < 1
    ):
        raise ValueError("La medida necesita tiempos positivos finitos y filas positivas")
    return dict(
        p50_ms=float(np.quantile(samples, 0.5)) * 1000,
        p95_ms=float(np.quantile(samples, 0.95)) * 1000,
        p99_ms=float(np.quantile(samples, 0.99)) * 1000,
        minimum_ms=min(samples) * 1000,
        maximum_ms=max(samples) * 1000,
        observations=rows * len(samples),
        rows_per_second=rows * len(samples) / sum(samples),
        samples_seconds=list(samples),
    )


def subset(raw, rows):
    """Copiar el mismo prefijo de flujos, sin completar ausencias ni fabricar activos."""
    if type(rows) is not int or not 1 <= rows <= len(raw["sample_ids"]):
        raise ValueError("El tamaño solicitado excede los flujos disponibles")
    return dict(
        inputs={name: value[:rows].copy() for name, value in raw["inputs"].items()},
        presence=raw["presence"][:rows].copy(),
        sample_ids=raw["sample_ids"][:rows],
        market=raw["market"][:rows],
        prediction_at=raw["prediction_at"][:rows].copy(),
        input_available_at=raw["input_available_at"][:rows].copy(),
    )


def passage(model, batches, *, graph):
    """Recorrer decisiones y, si se pide, derivar una pérdida sin aplicar los gradientes."""
    for parameter in model.parameters():
        parameter.grad = None
    torch.cuda.synchronize()
    start = time.perf_counter()
    state = model.initial_state(batches[0].flow_ids, differentiable=graph)
    context = _initial_context(model, differentiable=graph)
    torch.cuda.synchronize()
    initialization = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    allocated_before = torch.cuda.memory_allocated()
    samples, predictions = [], []
    with torch.set_grad_enabled(graph):
        for batch in batches:
            torch.cuda.synchronize()
            start = time.perf_counter()
            result, context = _step(model, batch, state, context, differentiable=graph)
            state = result.next_state
            predictions.append(result.quantiles)
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - start)
    backward = None
    if graph:
        start = time.perf_counter()
        quantiles = torch.cat(predictions)
        loss = pinball_loss(quantiles, torch.zeros_like(quantiles[:, 0]))
        loss.backward()
        torch.cuda.synchronize()
        backward = time.perf_counter() - start
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("El recorrido produjo un gradiente no finito")
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    # La salida solo se usa para paridad, no se compara con etiquetas financieras.
    prediction = torch.cat([x.detach() for x in predictions]).cpu()
    for parameter in model.parameters():
        parameter.grad = None
    return dict(
        initialization_seconds=initialization,
        decision_seconds=samples,
        backward_seconds=backward,
        allocated_before_bytes=allocated_before,
        peak_allocated_bytes=peak,
        peak_reserved_bytes=reserved,
    ), prediction


def measure(data, output, *, sizes, repetitions, warmup, dtype, declaration_path=None):
    if output.exists():
        raise FileExistsError("La medida necesita un archivo nuevo")
    if (
        not 2 <= repetitions <= 1000
        or not 1 <= warmup <= 32
        or not 1 <= len(sizes) <= 16
        or len(set(sizes)) != len(sizes)
    ):
        raise ValueError("Se necesitan repeticiones y calentamiento acotados y tamaños únicos")
    declaration_path = DECLARATION if declaration_path is None else Path(declaration_path)
    declaration = json.loads(declaration_path.read_text())
    parity = declaration.get("unity_control", ["cascade_original", "cascade_unity"])
    if (
        not isinstance(parity, list)
        or len(parity) != 2
        or parity[0] == parity[1]
        or any(name not in declaration["arms"] for name in parity)
        or not 1 <= len(declaration["arms"]) <= 16
    ):
        raise ValueError(
            "El catálogo necesita dos controles distintos y entre una y dieciséis configuraciones"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere cuda:0. No hay dispositivo CUDA disponible")
    torch.cuda.set_device("cuda:0")
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    metadata, specification, raws = load_inputs(data)
    recipe_path = ROOT / declaration["base_recipe"]
    recipe = json.loads(recipe_path.read_text())
    base = {key: value for key, value in recipe["predictor"].items() if key != "dtype"}
    device = torch.device("cuda:0")
    precision = getattr(torch, dtype)
    gpu = subprocess.check_output(["nvidia-smi"], text=True)
    cases = {}
    for size in sizes:
        transfer_start = time.perf_counter()
        batches = [
            DecisionBatch.from_corpus(
                subset(raw, size), specification, device=device, dtype=precision
            )
            for raw in raws
        ]
        torch.cuda.synchronize()
        input_seconds = time.perf_counter() - transfer_start
        if any(batch.flow_ids != batches[0].flow_ids for batch in batches):
            raise ValueError("Las sesiones materializadas no conservan los mismos flujos")
        built = {}
        for name, arm in declaration["arms"].items():
            config = FinancialConfig(
                specification,
                variant=declaration["variant"],
                seed=42,
                **(base | arm["predictor_changes"]),
            )
            model = FinancialPredictor(config, device=device, dtype=precision).train()
            built[name] = (model, _model_digest(model))
        records = {name: dict(inference=[], graph=[]) for name in built}
        names = list(built)
        unity = {}
        with unfused_attention():
            for repeat in range(warmup + repetitions):
                offset = repeat % len(names)
                for name in names[offset:] + names[:offset]:
                    for mode in ("inference", "graph"):
                        record, prediction = passage(built[name][0], batches, graph=mode == "graph")
                        if repeat >= warmup:
                            records[name][mode].append(record)
                        if repeat == warmup and mode == "inference" and name in parity:
                            unity[name] = prediction
        torch.testing.assert_close(unity[parity[0]], unity[parity[1]], rtol=0, atol=0)
        for name, (model, before) in built.items():
            if before != _model_digest(model):
                raise RuntimeError("La medida cambió parámetros compartidos")
            inference = records[name]["inference"]
            graph = records[name]["graph"]
            cases[f"{name}/{size}"] = dict(
                role=declaration["arms"][name]["role"],
                configuration=model.config.identity(),
                shared_parameters=sum(p.numel() for p in model.parameters()),
                state_tensor_bytes_per_flow=model._tensor_bytes_per_flow(),
                input_preparation_and_transfer_seconds=input_seconds,
                inference=summary([s for r in inference for s in r["decision_seconds"]], rows=size),
                forward_with_graph=summary(
                    [s for r in graph for s in r["decision_seconds"]], rows=size
                ),
                backward_including_loss=summary(
                    [r["backward_seconds"] for r in graph], rows=size * len(batches)
                ),
                passes=records[name],
                parameters_unchanged=True,
                unity_control_bitwise_parity=True,
            )
            print(
                f"{name}/{size}: {cases[f'{name}/{size}']['inference']['p50_ms']:.3f} ms",
                flush=True,
            )
        del built, model, batches, unity, prediction
        torch.cuda.empty_cache()
    receipt = dict(
        schema_version=1,
        kind=declaration.get("cost_kind", "neural_plasticity_cuda_cost"),
        measured_at_utc=datetime.now(UTC).isoformat(),
        code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        benchmark_sha256=sha256(Path(__file__)),
        declaration_sha256=sha256(declaration_path),
        recipe_sha256=sha256(recipe_path),
        inputs=dict(
            data_sha256=sha256(data),
            dimensions=metadata["dimensions"],
            view_manifest_sha256=metadata["view_manifest_sha256"],
            decisions=len(raws),
            targets_read=False,
        ),
        device="cuda:0",
        precision=dtype,
        tf32=False,
        versions=dict(
            python=platform.python_version(), torch=torch.__version__, cuda=torch.version.cuda
        ),
        gpu_at_start=gpu,
        threads=torch.get_num_threads(),
        load_average=os.getloadavg(),
        warmup=warmup,
        repetitions=repetitions,
        percentile_method="numpy.quantile, linear",
        resident_models=len(declaration["arms"]),
        timing_scope=(
            "DecisionBatch residente, prepare con validaciones y actualización neuronal. "
            "Inicialización y transferencia separadas."
        ),
        order="Alternancia de modelos con rotación por repetición",
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        unmeasured=[
            "Lectura Parquet por decisión y codificadores anteriores al corpus",
            "Energía",
            "Contadores internos de kernels y bytes físicos transferidos",
        ],
        optimizer_steps=0,
        scientific_evaluation=False,
        final_test_access=False,
        cases=cases,
    )
    atomic_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", nargs="+", type=int, default=[1, 64, 128])
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--declaration", type=Path)
    args = parser.parse_args()
    measure(
        args.data,
        args.output,
        sizes=args.sizes,
        repetitions=args.repetitions,
        warmup=args.warmup,
        dtype=args.dtype,
        declaration_path=args.declaration,
    )


if __name__ == "__main__":
    main()
