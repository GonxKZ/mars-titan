"""Emisión de Titans-MAC `mac_online` repetida con un CUDA Graph frente a la ruta eager.

Toma el primer evento guardado por `chronological_events.py` con al menos FILAS flujos,
construye el predictor de la receta de la campaña con la semilla 42 y mide un bloque
completo con `differentiable=True`, como emite el ajuste con acumulación:

- `prepare`, la emisión del ajuste, con sus validaciones y su comprobación inmediata en el
  codificador de precios,
- el mismo cálculo con todas las comprobaciones diferidas y una sola sincronización,
- ese cálculo capturado en un CUDA Graph, con copia de entradas, repetición, copia de
  salidas y lectura de las comprobaciones.

Exige los mismos bits en predicción, cuantiles, token, salida de la memoria y estado
siguiente de la memoria en las tres rutas. Solo mide: no hay backward ni optimizador. La
vista solo aporta su representación y su política de entradas, así que no se leen muestras.

Cada ruta repite sus medidas dentro de un rango NVTX con su nombre, para separarlas en un
perfil de Nsight Systems. REPETICIONES permite alargarlas para un perfil de muestreo.

Uso: `CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONPATH=src python benchmarks/titans_emission_graph.py
EVENTOS VISTA SALIDA [FILAS] [REPETICIONES]`
"""

import json
import pickle
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from mars_titan.models.baselines.inputs import MODALITIES
from mars_titan.models.baselines.transformer import nonfinite_flags
from mars_titan.models.quantile_head import median
from mars_titan.models.titans import state as titans_state
from mars_titan.models.titans.financial import gate_price_window
from mars_titan.models.titans.financial_inputs import (
    DecisionBatch,
    FinancialInputSpec,
    validated_cpu_batch,
)
from mars_titan.training import titans_walk_forward
from mars_titan.training.kernel_policy import apply_kernel_policy

RECIPE = Path("configs/titans/chronological-training-historical-masked.json")
WARMUP, REPEATS = 3, 20


def rows(raw, count):
    """Las primeras `count` filas de un bloque guardado, con la misma estructura."""
    if isinstance(raw, dict):
        return {key: rows(value, count) for key, value in raw.items()}
    if isinstance(raw, (np.ndarray, list, tuple)):
        return raw[:count]
    return raw


def first_block(events, count):
    with open(events, "rb") as handle:
        replay = pickle.load(handle)
    blocks = (block for event in replay["events"] for block in event.inputs)
    return rows(next(block for block in blocks if len(block["market"]) >= count), count)


def flat(point, quantiles, token, working, mac):
    memory = mac.memory
    return [
        point,
        quantiles,
        token,
        working,
        *memory.weights,
        *memory.momentum,
        memory.steps,
        *mac.convolution,
    ]


def median_seconds(function, label, repeats):
    for _ in range(WARMUP):
        function()
    torch.cuda.synchronize()
    times = []
    with torch.cuda.nvtx.range(label):
        for _ in range(repeats):
            began = time.perf_counter()
            function()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - began)
    return statistics.median(times)


def main(events, view, output, count=1024, repeats=REPEATS):
    apply_kernel_policy("fp32_strict")
    torch.use_deterministic_algorithms(True)
    raw = first_block(events, count)
    manifest = json.loads(Path(view).read_text())
    spec = FinancialInputSpec(
        source_sha256="0" * 64,
        view_sha256="1" * 64,
        representation=manifest["representation"],
        dimensions={name: raw["inputs"][name].shape[-1] for name in raw["inputs"]},
        input_policy=manifest["input_policy"],
    )
    cpu = validated_cpu_batch(raw, spec)
    document = json.loads(RECIPE.read_text())
    device = torch.device("cuda:0")
    predictor, _ = titans_walk_forward._predictor(document, spec, "mac_online", 42, device)
    predictor.train()
    batch = DecisionBatch.from_validated(cpu, device=device, dtype=torch.float32)
    state = predictor.initial_state(tuple(cpu.flow_ids), differentiable=True)

    def compute(inputs, presence, mac_state):
        # El cálculo de `prepare` con las comprobaciones del codificador de precios diferidas.
        price, checks = predictor.price_encoder.encode_last(gate_price_window(inputs["prices"]))
        flags = nonfinite_flags(checks)
        for index, (_, message) in enumerate(checks):
            titans_state.require_true(flags[index], message)
        representations = [price]
        for index, name in enumerate(MODALITIES[1:], 1):
            projected = predictor.encoders[name](inputs[name])
            if predictor.masked:
                projected = projected * presence[:, index : index + 1]
            representations.append(projected)
        if predictor.masked:
            representations.append(presence.to(dtype=predictor.head.weight.dtype))
        token = predictor.fusion(torch.cat(representations, dim=-1))
        titans_state.check_finite(token, "El token fusionado")
        encoded, next_mac = predictor.mac(token.unsqueeze(1), mac_state, differentiable=True)
        quantiles = predictor.head(encoded)
        titans_state.check_finite(quantiles, "Los cuantiles")
        return median(quantiles), quantiles, token.detach().clone(), encoded.clone(), next_mac

    def prepared():
        result = predictor.prepare(batch, state, differentiable=True)
        return flat(
            result.point_predictions,
            result.quantiles,
            result.detached_tokens,
            result.working_state,
            result.next_state.mac,
        )

    def deferred():
        with titans_state.deferred_checks(), torch.enable_grad():
            return flat(*compute(batch.inputs, batch.presence, state.mac))

    reference = [value.detach().clone() for value in prepared()]
    same_deferred = all(
        torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(reference, deferred(), strict=True)
    )
    # Captura con entradas en buffers fijos. Las comprobaciones se leen tras cada repetición.
    inputs = {name: value.clone() for name, value in batch.inputs.items()}
    presence = batch.presence.clone()
    pending = titans_state._PendingChecks()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(WARMUP):
            with titans_state.deferred_checks(), torch.enable_grad():
                compute(inputs, presence, state.mac)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    token = titans_state._PENDING.set(pending)
    try:
        with torch.cuda.graph(graph), torch.enable_grad():
            captured = flat(*compute(inputs, presence, state.mac))
    finally:
        titans_state._PENDING.reset(token)

    def graphed():
        for name, value in batch.inputs.items():
            inputs[name].copy_(value)
        presence.copy_(batch.presence)
        graph.replay()
        outputs = [value.clone() for value in captured]
        flags = torch.stack(pending.flags).tolist()
        if not all(flags):
            raise ValueError(pending.messages[flags.index(False)])
        return outputs

    same_graph = all(
        torch.equal(a, b) and a.dtype == b.dtype for a, b in zip(reference, graphed(), strict=True)
    )
    seconds = dict(
        prepare=median_seconds(prepared, "prepare", repeats),
        deferred=median_seconds(deferred, "deferred", repeats),
        graph=median_seconds(graphed, "graph", repeats),
    )
    report = dict(
        rows=count,
        recipe=str(RECIPE),
        variant="mac_online",
        differentiable=True,
        same_bits=dict(deferred_vs_prepare=same_deferred, graph_vs_prepare=same_graph),
        median_ms={name: round(value * 1e3, 3) for name, value in seconds.items()},
        graph_vs_deferred=round(seconds["deferred"] / seconds["graph"], 3),
        deferred_vs_prepare=round(seconds["prepare"] / seconds["deferred"], 3),
        deferred_checks=len(pending.flags),
        reserved_mib=round(torch.cuda.memory_reserved() / 2**20),
        warmup=WARMUP,
        repeats=repeats,
        torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),
    )
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not (same_deferred and same_graph):
        raise SystemExit("La emisión con grafo o diferida no repite los bits de prepare")


if __name__ == "__main__":
    main(*sys.argv[1:4], *(int(value) for value in sys.argv[4:6]))
