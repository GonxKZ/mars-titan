"""Coste del puente de tres factores P1 (#784) en la red H1 y en su regla por evento.

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo declarado
la llamada a `HolographicNetwork` de la referencia H1 de la declaración, sin pesos rápidos y
con ellos, en FP32 y con H = 64. Las entradas son las de la medida de H1: formas de la edición
v3.1 con noticias, fundamentales y macro ausentes en parte de las filas. Los pesos rápidos son
fijos, aleatorios y dentro de la cota, y la red devuelve sus trazas como en el recorrido.

`forward` es la inferencia sin gradiente. `backward` añade la propagación de una suma de
cuadrados hasta los parámetros de la red y de los codificadores, sin optimizador. Un gancho de
`torch.optim` rechaza cualquier paso. La lectura W_o y los ajustes U se fijan distintos de cero
para que el gradiente recorra el puente. Las dos pasadas invierten el orden de los modos y
comparten proceso con las aplicaciones que ya existan, así que los percentiles son
descriptivos.

La regla se mide aparte en CPU para cada configuración declarada con P1: guardar las trazas de
un evento con 64, 256 y 1.024 predicciones de un mercado y aplicar sus etiquetas, y leer los
pesos de esas filas. La salida debe ser nueva.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from benchmarks.holographic_network_cost import (
    CONCEPTS,
    INDICATORS,
    forbid_optimizer_steps,
    inputs,
    representation,
)
from mars_titan.data.storage import atomic_json

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs/experiments/three-factor-bridge-v1.json"
REFERENCE = "h1_reference"
MODES = ("h1_reference", "three_factor_bridge")
ROWS = (64, 256, 1024)
EVENT_ROWS = (64, 256, 1024)
HIDDEN, SEED, WARMUP, REPETITIONS = 64, 42, 3, 20
PASSES = ("forward", "backward")
# Pesos de cada arista en [−0,25, 0,25]: con dos aristas por área en el anillo, la suma de los
# que llegan a un área no pasa de 0,5 y queda dentro de la cota B = 1 de la declaración.
WEIGHT_SCALE = 0.25


def declared():
    """Configuraciones de la declaración, en su orden."""
    return json.loads(DECLARATION.read_text(encoding="utf-8"))["arms"]


def bridges():
    """Configuraciones que declaran P1, con su puente."""
    return {
        name: arm["predictor_changes"]["three_factor_bridge"]
        for name, arm in declared().items()
        if "three_factor_bridge" in arm["predictor_changes"]
    }


def protocol(device):
    return dict(
        kind="three_factor_bridge_cost_v1",
        device=device,
        declaration="configs/experiments/three-factor-bridge-v1.json",
        network=REFERENCE,
        modes=list(MODES),
        orders=[list(MODES), list(reversed(MODES))],
        rows=list(ROWS),
        hidden_size=HIDDEN,
        dtype="float32",
        passes=list(PASSES),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        fast_weights=f"fixed_uniform_within_plus_minus_{WEIGHT_SCALE}_and_traces_returned",
        readout="same_nonzero_readout_and_area_adapters_in_both_modes",
        rule=dict(
            configurations=list(bridges()),
            event_rows=list(EVENT_ROWS),
            device="cpu",
            measured="hold_and_settle_of_one_event_and_the_read_of_its_rows",
        ),
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def measure(module, encoders, arguments, kind, bias, synchronize):
    import torch

    blocks, presence, family = arguments

    def call():
        options = {} if bias is None else dict(edge_bias=bias, traces=[])
        if kind == "forward":
            with torch.no_grad():
                module(blocks, presence, family, encoders, **options)
        else:
            module.zero_grad(set_to_none=True)
            encoders.zero_grad(set_to_none=True)
            module(blocks, presence, family, encoders, **options).square().sum().backward()
        synchronize()

    for _ in range(WARMUP):
        call()
    samples = []
    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def _percentiles(values):
    pooled = sorted(values)
    return dict(
        p50=statistics.median(pooled),
        p95=pooled[min(len(pooled) - 1, round(0.95 * (len(pooled) - 1)))],
    )


def rule_cost(edges, areas):
    """Microsegundos por evento de la regla sola, en FP64 y CPU, para cada configuración."""
    import numpy as np

    from mars_titan.memory.three_factor_bridge import BridgePlasticity
    from mars_titan.models.titans.plastic_bridge import (
        MATURE_DIRECTIONAL_HIT,
        parse_three_factor_bridge,
    )

    result = {}
    for name, raw in bridges().items():
        config = parse_three_factor_bridge(raw)
        common = config.source == MATURE_DIRECTIONAL_HIT
        for count in EVENT_ROWS:
            flows = [f"US/S{asset:04}" for asset in range(count)]
            generator = np.random.default_rng(SEED + count)
            plastic = BridgePlasticity(config, edges, areas)
            event, read = [], []
            # El primer evento es el calentamiento y fija la expectativa del error.
            for repetition in range(WARMUP + REPETITIONS):
                keys = [(flow, repetition) for flow in flows]
                traces = generator.uniform(0, 0.2, size=(count, len(edges)))
                issued, outcomes = generator.normal(0, 0.02, size=(2, count)).tolist()
                labels = list(zip(flows, [repetition] * count, issued, outcomes, strict=True))
                expected = dict(US=float(generator.uniform(-0.5, 0.5))) if common else None
                start = time.perf_counter_ns()
                plastic.hold(keys, traces)
                plastic.settle(labels, expected)
                elapsed = (time.perf_counter_ns() - start) / 1e3
                start = time.perf_counter_ns()
                plastic.biases(flows)
                reading = (time.perf_counter_ns() - start) / 1e3
                if repetition >= WARMUP:
                    event.append(elapsed)
                    read.append(reading)
            result[f"{name}/{count}"] = dict(
                event_microseconds=dict(_percentiles(event), samples=event),
                read_microseconds=dict(_percentiles(read), samples=read),
            )
    return result


def run(device):
    import torch
    from torch import nn
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    from mars_titan.models.titans.holographic_network import (
        HolographicNetwork,
        parse_holographic_network,
    )

    if device.startswith("cuda") and (device != "cuda:0" or not torch.cuda.is_available()):
        raise RuntimeError("La medida CUDA requiere cuda:0 disponible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cuda = device.startswith("cuda")
    synchronize = torch.cuda.synchronize if cuda else lambda: None
    guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
    config = parse_holographic_network(
        declared()[REFERENCE]["predictor_changes"]["holographic_network"]
    )
    module = HolographicNetwork(
        HIDDEN, config, representation(), seed=SEED, max_batch=max(ROWS)
    ).to(device)
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        module.readout.weight.copy_(
            (torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5).to(device)
        )
        module.area_up.copy_(torch.randn(module.area_up.shape, generator=generator).to(device) / 4)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        encoders = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(3 * width, HIDDEN), nn.SiLU())
                for name, width in (("fundamentals", CONCEPTS), ("macro", INDICATORS))
            }
        ).to(device)
    edges = module.plastic_edges
    batches = {rows: inputs(rows, device) for rows in ROWS}
    weights = {
        rows: ((torch.rand(rows, len(edges), generator=generator) * 2 - 1) * WEIGHT_SCALE).to(
            device
        )
        for rows in ROWS
    }
    records, peaks = [], {}
    try:
        for order in (MODES, tuple(reversed(MODES))):
            for rows in ROWS:
                for kind in PASSES:
                    for mode in order:
                        bias = None if mode == REFERENCE else weights[rows]
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                        samples = measure(module, encoders, batches[rows], kind, bias, synchronize)
                        records.append(dict(mode=mode, rows=rows, pass_=kind, ms=samples))
                        if cuda:
                            key = f"{mode}/{rows}/{kind}"
                            peaks[key] = max(peaks.get(key, 0), torch.cuda.max_memory_allocated())
        rule = rule_cost(edges, config.catalog_areas)
    finally:
        guard.remove()
    summary = []
    for mode in MODES:
        for rows in ROWS:
            for kind in PASSES:
                pooled = [
                    value
                    for record in records
                    if (record["mode"], record["rows"], record["pass_"]) == (mode, rows, kind)
                    for value in record["ms"]
                ]
                percentiles = _percentiles(pooled)
                summary.append(
                    dict(
                        mode=mode,
                        rows=rows,
                        pass_=kind,
                        p50_ms=percentiles["p50"],
                        p95_ms=percentiles["p95"],
                        peak_cuda_bytes=peaks.get(f"{mode}/{rows}/{kind}"),
                    )
                )
    return dict(
        protocol=protocol(device),
        finished_at=datetime.now(UTC).isoformat(),
        environment=dict(
            python=sys.version.split()[0],
            torch=torch.__version__,
            machine=platform.machine(),
            processor=platform.processor(),
            threads=torch.get_num_threads(),
            gpu=torch.cuda.get_device_name(0) if cuda else None,
        ),
        # P1 no añade parámetros: los pesos rápidos son estado del recorrido.
        parameters=sum(value.numel() for value in module.parameters()),
        edges=len(edges),
        # Estado en FP64: los pesos de cada mercado y la traza de cada predicción pendiente.
        state_bytes=dict(
            market_weights=8 * len(edges), pending_trace_per_prediction=8 * len(edges)
        ),
        memory="peak_cuda_bytes_per_mode_rows_and_pass" if cuda else "not_measured_on_cpu",
        summary=summary,
        records=records,
        rule=rule,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()
    if not arguments.run:
        print(json.dumps(protocol(arguments.device), indent=2))
        return
    if arguments.output is None or arguments.output.exists():
        raise SystemExit("La medida necesita --output con un destino nuevo")
    # atomic_json crea la carpeta, escribe en un temporal y lo confirma con os.replace.
    atomic_json(arguments.output, run(arguments.device))


if __name__ == "__main__":
    main()
