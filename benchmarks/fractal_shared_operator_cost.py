"""Coste del operador fractal C3 y de sus cuatro controles (#751).

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo declarado
la llamada a `FractalSharedOperator` de cada modelo de la declaración, en FP32 y con H = 64.
Las entradas son sintéticas con las formas de la emisión: cinco bloques [R,5,H] con noticias,
fundamentales y macro ausentes en parte de las filas y ventanas de 64 sesiones con seis
canales, cuyos cierres cargan en ocho factores. Con 64 y 256 filas el bloque es una sesión.
Con 1.024 son cuatro sesiones de 256 activos, en dos mercados y dos instantes.

`sectors` mide solo el reparto de las sesiones, con la copia de la ventana a la CPU y el
agrupamiento. `forward` es la inferencia sin gradiente, reparto incluido. `backward` añade la
propagación de una suma de cuadrados hasta los parámetros del módulo, sin optimizador. Un
gancho de `torch.optim` rechaza cualquier paso. Las proyecciones de salida se fijan distintas
de cero. Las dos pasadas invierten el orden de los modelos y comparten proceso con las
aplicaciones que ya existan, así que los percentiles son descriptivos. La salida debe ser
nueva.
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
DECLARATION = ROOT / "configs/experiments/fractal-shared-operator-v1.json"
ROWS = (64, 256, 1024)
# Activos por sesión de cada tamaño de bloque.
SESSION_ASSETS = {64: 64, 256: 256, 1024: 256}
HIDDEN, SEED, WARMUP, REPETITIONS, FACTORS = 64, 42, 3, 20, 8
PASSES = ("sectors", "forward", "backward")
# Fracción de filas con cada modalidad opcional, como en las medidas de F1, F5, C2 y H1.
OBSERVED = dict(news=0.4, fundamentals=0.7, macro=0.9)


def arms():
    """Modelos de la declaración que añaden C3, en su orden."""
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    return {
        name: arm["predictor_changes"]["fractal_shared_operator"]
        for name, arm in declaration["arms"].items()
        if arm["predictor_changes"]
    }


def protocol(device):
    modes = list(arms())
    return dict(
        kind="fractal_shared_operator_cost_v1",
        device=device,
        declaration="configs/experiments/fractal-shared-operator-v1.json",
        modes=modes,
        orders=[modes, list(reversed(modes))],
        rows=list(ROWS),
        session_assets={str(rows): assets for rows, assets in SESSION_ASSETS.items()},
        hidden_size=HIDDEN,
        dtype="float32",
        inputs=dict(
            observed_rows=OBSERVED,
            prices="64_sessions_six_channels_closes_loaded_on_eight_factors",
        ),
        passes=list(PASSES),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        projections="nonzero_output_projections_in_every_arm",
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La medida de coste no admite pasos de optimizador")


def inputs(rows, device):
    """Bloques, presencia, flujos, instantes y ventanas de un bloque de R filas."""
    import numpy as np
    import torch

    generator = np.random.default_rng(SEED + rows)
    presence = np.ones((rows, 5), dtype=np.bool_)
    for index, name in ((1, "news"), (3, "fundamentals"), (4, "macro")):
        presence[:, index] = generator.random(rows) < OBSERVED[name]
    blocks = generator.normal(size=(rows, 5, HIDDEN)).astype(np.float32) * presence[..., None]
    assets = SESSION_ASSETS[rows]
    sessions = rows // assets
    markets = ("US", "CN")
    flows = tuple(
        f"{markets[row // assets % 2]}/A{row // assets // 2}{row % assets:04d}"
        for row in range(rows)
    )
    instants = tuple(7 + row // (2 * assets) for row in range(rows))
    loading = generator.integers(FACTORS, size=rows)
    common = generator.normal(size=(sessions, FACTORS, 63))
    returns = common[np.arange(rows) // assets, loading] + generator.normal(size=(rows, 63))
    close = np.concatenate([np.zeros((rows, 1)), np.cumsum(0.01 * returns, axis=1)], axis=1)
    window = np.zeros((rows, 64, 6), dtype=np.float32)
    window[..., 3] = close
    window[..., :3] = close[..., None] + 0.002 * generator.normal(size=(rows, 64, 3))
    window[..., 4] = generator.normal(size=(rows, 64))
    window[..., 5] = 1.0
    return (
        torch.from_numpy(blocks).to(device),
        torch.from_numpy(presence).to(device),
        flows,
        instants,
        torch.from_numpy(window).to(device),
    )


def measure(module, arguments, kind, synchronize):
    import torch

    blocks, presence, flows, instants, prices = arguments

    def call():
        if kind == "sectors":
            module.sessions(flows, instants, prices)
        elif kind == "forward":
            with torch.no_grad():
                module(blocks, presence, flows, instants, prices)
        else:
            module.zero_grad(set_to_none=True)
            module(blocks, presence, flows, instants, prices).square().sum().backward()
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

    from mars_titan.models.titans.fractal_shared_operator import (
        PER_LEVEL,
        FractalSharedOperator,
        parse_fractal_shared_operator,
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
        name: FractalSharedOperator(HIDDEN, parse_fractal_shared_operator(raw), seed=SEED).to(
            device
        )
        for name, raw in declared.items()
    }
    generator = torch.Generator().manual_seed(SEED)
    output = torch.randn(HIDDEN, HIDDEN, generator=generator) / 4
    for module in modules.values():
        with torch.no_grad():
            if module.config.kind == PER_LEVEL:
                module.left[:, -1].copy_(output[:, : module.control_rank].to(device))
            else:
                module.weight[-1].copy_(output.to(device))
    batches = {rows: inputs(rows, device) for rows in ROWS}
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
    # La carpeta se crea antes de medir, para no perder una medida al escribirla.
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    result = run(arguments.device)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
