"""Coste de la atención axial C2 y de su control de solo tiempo (#751).

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo declarado
la llamada a `AxialTimeRegion` de cada modelo de la declaración, en FP32 y con H = 64. Las
entradas son sintéticas con las formas de la emisión: cinco bloques [R,5,H] con noticias,
fundamentales y macro ausentes en parte de las filas y un historial lleno de W − 1 = 7
emisiones con las mismas fracciones de presencia.

`forward` es la inferencia sin gradiente, incluido el desplazamiento del historial.
`backward` añade la propagación de una suma de cuadrados hasta los parámetros del módulo, sin
optimizador. Un gancho de `torch.optim` rechaza cualquier paso. Las dos proyecciones de
salida se fijan iguales y distintas de cero en los dos modelos. Las dos pasadas invierten el
orden de los modelos y comparten proceso con las aplicaciones que ya existan, así que los
percentiles son descriptivos. La salida debe ser nueva.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs/experiments/axial-time-region-v1.json"
ROWS = (64, 256, 1024)
HIDDEN, SEED, WARMUP, REPETITIONS = 64, 42, 3, 20
PASSES = ("forward", "backward")
# Fracción de filas con cada modalidad opcional, como en las medidas de F1, F5 y H1.
OBSERVED = dict(news=0.4, fundamentals=0.7, macro=0.9)


def arms():
    """Modelos de la declaración que añaden C2, en su orden."""
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    return {
        name: arm["predictor_changes"]["axial_time_region"]
        for name, arm in declaration["arms"].items()
        if arm["predictor_changes"]
    }


def protocol(device):
    modes = list(arms())
    return dict(
        kind="axial_time_region_cost_v1",
        device=device,
        declaration="configs/experiments/axial-time-region-v1.json",
        modes=modes,
        orders=[modes, list(reversed(modes))],
        rows=list(ROWS),
        hidden_size=HIDDEN,
        dtype="float32",
        inputs=dict(observed_rows=OBSERVED, history="full_with_the_same_presence_fractions"),
        passes=list(PASSES),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        projections="same_nonzero_output_projections_in_both_arms",
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La medida de coste no admite pasos de optimizador")


def inputs(rows, window, device):
    """Bloques, presencia e historial de un bloque de R filas."""
    import numpy as np
    import torch

    from mars_titan.models.titans.axial_time_region import RegionHistory

    generator = np.random.default_rng(SEED + rows)

    def presence(*shape):
        result = np.ones((*shape, 5), dtype=np.bool_)
        for index, name in ((1, "news"), (3, "fundamentals"), (4, "macro")):
            result[..., index] = generator.random(shape) < OBSERVED[name]
        return result

    current = presence(rows)
    past = presence(rows, window - 1)
    blocks = generator.normal(size=(rows, 5, HIDDEN)).astype(np.float32) * current[..., None]
    history = generator.normal(size=(rows, window - 1, 5, HIDDEN)).astype(np.float32)
    history *= past[..., None]
    return (
        torch.from_numpy(blocks).to(device),
        torch.from_numpy(current).to(device),
        RegionHistory(torch.from_numpy(history).to(device), torch.from_numpy(past).to(device)),
    )


def measure(module, arguments, kind, synchronize):
    import torch

    blocks, presence, history = arguments

    def call():
        if kind == "forward":
            with torch.no_grad():
                module(blocks, presence, history)
        else:
            module.zero_grad(set_to_none=True)
            module(blocks, presence, history)[0].square().sum().backward()
        synchronize()

    for _ in range(WARMUP):
        call()
    samples = []
    for _ in range(REPETITIONS):
        start = time.perf_counter_ns()
        call()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return samples


def run(device):
    import torch
    from torch.optim.optimizer import register_optimizer_step_pre_hook

    from mars_titan.models.titans.axial_time_region import (
        AxialTimeRegion,
        parse_axial_time_region,
    )

    if device.startswith("cuda") and (device != "cuda:0" or not torch.cuda.is_available()):
        raise RuntimeError("La medida CUDA requiere cuda:0 disponible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cuda = device.startswith("cuda")
    synchronize = torch.cuda.synchronize if cuda else lambda: None
    guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
    declared = arms()
    modules = {
        name: AxialTimeRegion(HIDDEN, parse_axial_time_region(raw), seed=SEED).to(device)
        for name, raw in declared.items()
    }
    generator = torch.Generator().manual_seed(SEED)
    projections = [torch.randn(HIDDEN, HIDDEN, generator=generator) / 4 for _ in range(2)]
    for module in modules.values():
        with torch.no_grad():
            module.time_out.weight.copy_(projections[0].to(device))
            module.second_out.weight.copy_(projections[1].to(device))
    windows = {module.config.window for module in modules.values()}
    if len(windows) != 1:
        raise RuntimeError("Los modelos de C2 deben compartir la ventana")
    window = windows.pop()
    batches = {rows: inputs(rows, window, device) for rows in ROWS}
    records, peaks = [], {}
    try:
        for order in (tuple(declared), tuple(reversed(declared))):
            for rows in ROWS:
                for kind in PASSES:
                    for mode in order:
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                        samples = measure(modules[mode], batches[rows], kind, synchronize)
                        records.append(dict(mode=mode, rows=rows, pass_=kind, ms=samples))
                        if cuda:
                            key = f"{mode}/{rows}/{kind}"
                            peaks[key] = max(peaks.get(key, 0), torch.cuda.max_memory_allocated())
    finally:
        guard.remove()
    summary = []
    for mode in declared:
        for rows in ROWS:
            for kind in PASSES:
                pooled = sorted(
                    value
                    for record in records
                    if (record["mode"], record["rows"], record["pass_"]) == (mode, rows, kind)
                    for value in record["ms"]
                )
                summary.append(
                    dict(
                        mode=mode,
                        rows=rows,
                        pass_=kind,
                        p50_ms=statistics.median(pooled),
                        p95_ms=pooled[min(len(pooled) - 1, round(0.95 * (len(pooled) - 1)))],
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
        parameters={
            mode: sum(value.numel() for value in module.parameters())
            for mode, module in modules.items()
        },
        history_bytes_per_flow={mode: module.bytes_per_flow() for mode, module in modules.items()},
        memory="peak_cuda_bytes_per_mode_rows_and_pass" if cuda else "not_measured_on_cpu",
        summary=summary,
        records=records,
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
    result = run(arguments.device)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
