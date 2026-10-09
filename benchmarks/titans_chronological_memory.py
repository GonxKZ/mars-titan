"""Medir el grafo que conserva el recorrido cronológico de Titans-MAC antes del backward.

Reproduce un tramo diferenciable del entrenador con la configuración de la receta y las
dimensiones reales de la edición histórica: precios 64×5, noticias 384, gráficos 512,
fundamentales 45 (15 conceptos) y macro 420 (140 indicadores). Las entradas son
aleatorias. Calcula pérdidas y gradientes para medir tiempos, sin optimizador ni pasos.
En `cuda:0` registra además el pico del asignador antes y después del backward.
"""

import argparse
import json
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.storage import atomic_json
from mars_titan.models.quantile_head import QUANTILE_HEAD, pinball_loss
from mars_titan.models.titans.financial import VARIANTS, FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import (
    DecisionBatch,
    FinancialInputSpec,
    validated_cpu_batch,
)
from mars_titan.training.graph_memory import saved_graph_bytes

DIMENSIONS = dict(prices=5, news=384, charts=512, fundamentals=45, macro=420)
CATALOGS = dict(fundamentals=15, macro=140)
# Activos con precios en la edición histórica y en su mayor mercado.
POPULATIONS = {"128": 128, "1024": 1024, "us_4202": 4202, "all_5023": 5023}
DAY_US = 86_400_000_000


def specification():
    representation = dict(
        **policy_identity(HISTORICAL_MASKED),
        fundamental_concepts=[f"concept_{i}" for i in range(CATALOGS["fundamentals"])],
        macro_indicators=[f"indicator_{i}" for i in range(CATALOGS["macro"])],
        encoders={"benchmark": "random_inputs"},
        representation_code={"benchmark.py": "0" * 64},
        text_aggregation="mean",
        context_sessions=64,
        news_lookback_sessions=5,
    )
    return FinancialInputSpec(
        source_sha256="a" * 64,
        view_sha256="b" * 64,
        representation=representation,
        dimensions=DIMENSIONS,
        input_policy=HISTORICAL_MASKED,
    )


def batch(spec, flows, at, generator, dtype, device):
    size = len(flows)
    values = {
        name: generator.normal(size=(size, 64, width) if name == "prices" else (size, width))
        for name, width in DIMENSIONS.items()
    }
    for name, width in CATALOGS.items():
        values[name][:, width : 2 * width] = 1
        values[name][:, 2 * width :] = np.abs(values[name][:, 2 * width :])
    raw = dict(
        inputs={name: value.astype(np.float32) for name, value in values.items()},
        presence=np.ones((size, 5), dtype=np.bool_),
        sample_ids=[f"{flow}/{at}" for flow in flows],
        market=["US"] * size,
        prediction_at=np.full(size, at, dtype="datetime64[us]"),
        input_available_at=np.full(size, at - 1, dtype="datetime64[us]"),
    )
    return DecisionBatch.from_validated(validated_cpu_batch(raw, spec), device=device, dtype=dtype)


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def segment(document, variant, rows, instants, dtype, device):
    """Un tramo del recorrido: el estado y las salidas conservan su grafo hasta el final."""
    spec = specification()
    options = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    model = FinancialPredictor(
        FinancialConfig(spec, variant=variant, seed=42, **options), device=device, dtype=dtype
    )
    model.train()
    flows = tuple(f"US/A{index:04d}" for index in range(rows))
    generator = np.random.default_rng(5)
    state = model.initial_state(flows, differentiable=True)
    quantiles = model.config.head == QUANTILE_HEAD
    outputs, input_bytes, cuda = [], 0, {}
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        cuda["allocated_before"] = torch.cuda.memory_allocated(device)
    started = time.perf_counter()
    for instant in range(instants):
        at = 1_609_459_200_000_000 + instant * DAY_US
        decision = batch(spec, flows, at, generator, dtype, device)
        input_bytes += sum(v.nbytes for v in (*decision.inputs.values(), decision.presence))
        prepared = model.prepare(decision, state, differentiable=True)
        state = prepared.next_state
        outputs.append(prepared.quantiles if quantiles else prepared.point_predictions)
    synchronize(device)
    forward = time.perf_counter() - started
    if cuda:
        cuda["allocated_before_backward"] = torch.cuda.memory_allocated(device)
        cuda["peak_before_backward"] = torch.cuda.max_memory_allocated(device)
    fast = [] if state.mac is None else [*state.mac.memory.weights, *state.mac.memory.momentum]
    parameters = list(model.parameters())
    graph = saved_graph_bytes([*outputs, *fast], exclude=parameters)
    prediction = torch.cat(outputs)
    target = torch.zeros(len(prediction), dtype=dtype, device=device)
    started = time.perf_counter()
    if quantiles:
        pinball_loss(prediction, target).backward()
    else:
        torch.nn.functional.l1_loss(prediction, target).backward()
    synchronize(device)
    backward = time.perf_counter() - started
    if cuda:
        cuda["peak_with_backward"] = torch.cuda.max_memory_allocated(device)
    return dict(
        rows=rows,
        instants=instants,
        saved_graph_bytes=graph,
        saved_bytes_per_row_instant=graph / (rows * instants),
        fast_state_bytes_per_flow=sum(t.nbytes for t in fast) / rows,
        input_bytes_per_row_instant=input_bytes / (rows * instants),
        parameter_bytes=sum(p.nbytes for p in parameters),
        forward_with_graph_seconds=forward,
        backward_seconds=backward,
        cuda_allocator_bytes=cuda or None,
    )


def estimates(measure, truncation, accumulation_rows):
    """Bytes vivos del tramo con el grafo completo y con bloques de flujos."""
    per_row = measure["saved_bytes_per_row_instant"]
    state, inputs = measure["fast_state_bytes_per_flow"], measure["input_bytes_per_row_instant"]
    result = {}
    for name, flows in POPULATIONS.items():
        block = min(flows, accumulation_rows)
        result[name] = dict(
            flows=flows,
            graph_bytes_per_event=per_row * flows,
            graph_bytes_per_segment=per_row * flows * truncation,
            # Grafo de un bloque, lotes del tramo y estados inicial y vigente de cada flujo.
            accumulated_bytes_per_segment=per_row * block * truncation
            + inputs * flows * truncation
            + 2 * state * flows,
        )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--recipe", type=Path, default=Path("configs/titans/chronological-training-quantile.json")
    )
    parser.add_argument("--rows", type=int, default=128)
    parser.add_argument("--accumulation-rows", type=int, default=128)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.backends.mha.set_fastpath_enabled(False)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó cuda:0 y no hay dispositivo CUDA visible")
    document = json.loads(args.recipe.read_text())
    dtype = getattr(torch, document["predictor"]["dtype"])
    truncation = document["recipe"]["truncation"]
    started = time.perf_counter()
    variants = {}
    for variant in VARIANTS:
        measure = segment(document, variant, args.rows, truncation, dtype, device)
        variants[variant] = dict(
            measure=measure,
            estimates=estimates(measure, truncation, args.accumulation_rows),
        )
        print(variant, round(measure["saved_bytes_per_row_instant"]), flush=True)
    linearity = {
        str(rows): segment(document, "mac_online", rows, truncation, dtype, device)[
            "saved_bytes_per_row_instant"
        ]
        for rows in (args.rows // 2, args.rows * 2)
    }
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "command": (
            "uv run --no-sync python benchmarks/titans_chronological_memory.py "
            f"--recipe {args.recipe} --device {args.device} --rows {args.rows} "
            f"--accumulation-rows {args.accumulation_rows} --output <recibo>"
        ),
        "environment": "CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2"
        if device.type == "cpu"
        else "cuda:0",
        "scope": (
            "Grafo guardado de un tramo con entradas aleatorias. No es un perfil del "
            "recorrido completo. Solo cuda:0 registra el asignador."
        ),
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "threads": args.threads,
        "recipe": str(args.recipe),
        "predictor": document["predictor"],
        "truncation": truncation,
        "dimensions": DIMENSIONS,
        "measure": (
            "Almacenamientos únicos guardados para el backward y alcanzables desde las "
            "salidas y el estado rápido del tramo, sin parámetros"
        ),
        "variants": variants,
        "mac_online_bytes_per_row_instant_by_rows": linearity,
        "accumulation_rows": args.accumulation_rows,
        "populations": POPULATIONS,
        "optimizer_steps": 0,
        "training_runs": 0,
        "gpu_workloads": int(device.type == "cuda"),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()
