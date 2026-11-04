"""Coste de C1 y C6 frente al operador regional que sustituyen, sin aprendizaje (#751).

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo
declarado la llamada al operador regional con bloques aleatorios [B,5,H] en FP32. Compara
`regional_fusion_v1` en anillo con resumen, C1 con su control sin mezcla y C6 con su
control de escritura densa, todos con el mismo calendario de ciclos y pasos.

Hay dos recorridos. `forward` es la inferencia sin gradiente. `backward` añade la
propagación de una suma de cuadrados hasta los parámetros del operador, sin optimizador.
Un gancho de `torch.optim` rechaza cualquier paso. Las dos pasadas invierten el orden de
las variantes y comparten proceso con las aplicaciones que ya existan, así que los
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

from mars_titan.data.storage import atomic_json

VARIANTS = ("regional_fusion", "c1_mixing", "c1_control", "c6_competition", "c6_dense")
ROWS = (64, 256)
HIDDEN, SEED, WARMUP, REPETITIONS = 64, 42, 3, 20
MODES = ("forward", "backward")


def protocol(device):
    return dict(
        kind="regional_variants_operator_cost_v1",
        device=device,
        variants=list(VARIANTS),
        orders=[list(VARIANTS), list(reversed(VARIANTS))],
        rows=list(ROWS),
        hidden_size=HIDDEN,
        dtype="float32",
        operator="ring, cycles=2, local_steps=2, summary=true, gains 0.25",
        presence="all five sources present",
        modes=list(MODES),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def build(name, operator):
    from mars_titan.models.titans.regional_fusion import RegionalFusion
    from mars_titan.models.titans.regional_variants import (
        RegionAxisMixingConfig,
        SharedWorkspaceConfig,
        build_regional_variant,
    )

    if name == "regional_fusion":
        return RegionalFusion(HIDDEN, operator, seed=SEED, max_batch=max(ROWS))
    config = dict(
        c1_mixing=RegionAxisMixingConfig(operator),
        c1_control=RegionAxisMixingConfig(operator, mixing="none"),
        c6_competition=SharedWorkspaceConfig(operator),
        c6_dense=SharedWorkspaceConfig(operator, write="dense", write_top_k=None),
    )[name]
    return build_regional_variant(HIDDEN, config, seed=SEED, max_batch=max(ROWS))


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La medida de coste no admite pasos de optimizador")


def measure(module, blocks, mode, synchronize):
    import torch

    def call():
        if mode == "forward":
            with torch.no_grad():
                module(blocks, None)
        else:
            module.zero_grad(set_to_none=True)
            module(blocks, None).square().sum().backward()
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

    from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

    if device.startswith("cuda") and (device != "cuda:0" or not torch.cuda.is_available()):
        raise RuntimeError("La medida CUDA requiere cuda:0 disponible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    synchronize = torch.cuda.synchronize if device.startswith("cuda") else lambda: None
    guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
    operator = RegionalFusionConfig(topology="ring")
    modules = {name: build(name, operator).to(device) for name in VARIANTS}
    generator = torch.Generator().manual_seed(SEED)
    inputs = {rows: torch.randn(rows, 5, HIDDEN, generator=generator).to(device) for rows in ROWS}
    records = []
    try:
        for order in (VARIANTS, tuple(reversed(VARIANTS))):
            for rows in ROWS:
                for mode in MODES:
                    for name in order:
                        samples = measure(modules[name], inputs[rows], mode, synchronize)
                        records.append(dict(variant=name, rows=rows, mode=mode, ms=samples))
    finally:
        guard.remove()
    summary = []
    for name in VARIANTS:
        for rows in ROWS:
            for mode in MODES:
                pooled = sorted(
                    value
                    for record in records
                    if (record["variant"], record["rows"], record["mode"]) == (name, rows, mode)
                    for value in record["ms"]
                )
                summary.append(
                    dict(
                        variant=name,
                        rows=rows,
                        mode=mode,
                        p50_ms=statistics.median(pooled),
                        p95_ms=pooled[min(len(pooled) - 1, round(0.95 * (len(pooled) - 1)))],
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
            gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        ),
        parameters={
            name: sum(value.numel() for value in modules[name].parameters()) for name in VARIANTS
        },
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
    # atomic_json crea la carpeta, escribe en un temporal y lo confirma con os.replace.
    atomic_json(arguments.output, run(arguments.device))


if __name__ == "__main__":
    main()
