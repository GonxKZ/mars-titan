"""Coste de la red holográfica H1 y de sus ablaciones H2 y H3 (#783).

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo declarado
la llamada a `HolographicNetwork` de cada brazo de la declaración, en FP32 y con H = 64. Las
entradas son sintéticas con las formas de la edición v3.1: 26 conceptos contables y 140
indicadores macro con valor, máscara y edad, y cinco bloques [R,5,H] con noticias,
fundamentales y macro ausentes en parte de las filas. Los codificadores de fundamentales y
macro son Linear + SiLU, como en la referencia.

`forward` es la inferencia sin gradiente. `backward` añade la propagación de una suma de
cuadrados hasta los parámetros de la red y de los codificadores, sin optimizador. Un gancho
de `torch.optim` rechaza cualquier paso. La lectura W_o y los ajustes U se fijan iguales y
distintos de cero en todos los brazos, para que el gradiente recorra el puente. Las dos
pasadas invierten el orden de los brazos y comparten proceso con las aplicaciones que ya
existan, así que los percentiles son descriptivos. La salida debe ser nueva.
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
DECLARATION = ROOT / "configs/experiments/holographic-network-v1.json"
ROWS = (64, 256, 1024)
HIDDEN, SEED, WARMUP, REPETITIONS = 64, 42, 3, 20
PASSES = ("forward", "backward")
CONCEPTS, INDICATORS = 26, 140
# Fracción de filas con cada modalidad opcional, como en las medidas de F1 y F5.
OBSERVED = dict(news=0.4, fundamentals=0.7, macro=0.9)


def arms():
    """Brazos de la declaración que añaden la red, en su orden."""
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    return {
        name: arm["predictor_changes"]["holographic_network"]
        for name, arm in declaration["arms"].items()
        if arm["predictor_changes"]
    }


def protocol(device):
    modes = list(arms())
    return dict(
        kind="holographic_network_cost_v1",
        device=device,
        declaration="configs/experiments/holographic-network-v1.json",
        modes=modes,
        orders=[modes, list(reversed(modes))],
        rows=list(ROWS),
        hidden_size=HIDDEN,
        dtype="float32",
        inputs=dict(
            fundamentals=3 * CONCEPTS,
            macro=3 * INDICATORS,
            layout="values_observed_log_age",
            observed_rows=OBSERVED,
        ),
        passes=list(PASSES),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        readout="same_nonzero_readout_and_area_adapters_in_every_arm",
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La medida de coste no admite pasos de optimizador")


def representation():
    from mars_titan.data.accounting_catalog import JOINT_CONCEPTS
    from mars_titan.models.titans.holographic_network import macro_categories

    return dict(
        fundamental_concepts=list(JOINT_CONCEPTS), macro_indicators=sorted(macro_categories())
    )


def masked(generator, rows, concepts, present):
    import numpy as np

    observed = (generator.random((rows, concepts)) < 0.6) & present[:, None]
    observed[present, 0] = True
    values = np.where(observed, generator.normal(size=(rows, concepts)), 0)
    ages = np.where(observed, np.log1p(generator.uniform(0, 400, size=(rows, concepts))), 0)
    return np.concatenate([values, observed, ages], axis=1).astype(np.float32)


def inputs(rows, device):
    """Bloques, presencia y entradas de familia de un bloque de R filas."""
    import numpy as np
    import torch

    generator = np.random.default_rng(SEED + rows)
    presence = np.ones((rows, 5), dtype=np.bool_)
    for index, name in ((1, "news"), (3, "fundamentals"), (4, "macro")):
        presence[:, index] = generator.random(rows) < OBSERVED[name]
    blocks = generator.normal(size=(rows, 5, HIDDEN)).astype(np.float32) * presence[..., None]
    family = dict(
        fundamentals=masked(generator, rows, CONCEPTS, presence[:, 3]),
        macro=masked(generator, rows, INDICATORS, presence[:, 4]),
    )
    return (
        torch.from_numpy(blocks).to(device),
        torch.from_numpy(presence).to(device),
        {name: torch.from_numpy(value).to(device) for name, value in family.items()},
    )


def measure(module, encoders, arguments, kind, synchronize):
    import torch

    blocks, presence, family = arguments

    def call():
        if kind == "forward":
            with torch.no_grad():
                module(blocks, presence, family, encoders)
        else:
            module.zero_grad(set_to_none=True)
            encoders.zero_grad(set_to_none=True)
            module(blocks, presence, family, encoders).square().sum().backward()
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
    layout = representation()
    declared = arms()
    modules = {
        name: HolographicNetwork(
            HIDDEN, parse_holographic_network(raw), layout, seed=SEED, max_batch=max(ROWS)
        ).to(device)
        for name, raw in declared.items()
    }
    generator = torch.Generator().manual_seed(SEED)
    readout = torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
    for module in modules.values():
        with torch.no_grad():
            module.readout.weight.copy_(readout.to(device))
            if module.config.catalog_areas:
                module.area_up.copy_(
                    torch.randn(module.area_up.shape, generator=generator).to(device) / 4
                )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        encoders = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(3 * width, HIDDEN), nn.SiLU())
                for name, width in (("fundamentals", CONCEPTS), ("macro", INDICATORS))
            }
        ).to(device)
    batches = {rows: inputs(rows, device) for rows in ROWS}
    records, peaks = [], {}
    try:
        for order in (tuple(declared), tuple(reversed(declared))):
            for rows in ROWS:
                for kind in PASSES:
                    for mode in order:
                        if cuda:
                            torch.cuda.reset_peak_memory_stats()
                        samples = measure(modules[mode], encoders, batches[rows], kind, synchronize)
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
    result = run(arguments.device)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
