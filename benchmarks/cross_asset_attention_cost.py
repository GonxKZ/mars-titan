"""Coste de la atención entre activos C5 y de su control sin aprendizaje (#751).

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo
declarado la llamada a `CrossAssetAttention` con representaciones aleatorias [R,H] en FP32.
Compara `same_session` con el control `self_only`, que calcula la misma matriz y la enmascara.
Los bloques reparten las filas en sesiones de igual tamaño, con dos mercados y el mismo
instante, como un bloque de decisión completo.

`forward` es la inferencia sin gradiente. `backward` añade la propagación de una suma de
cuadrados hasta los parámetros de C5, sin optimizador. Un gancho de `torch.optim` rechaza
cualquier paso. Las dos pasadas invierten el orden de los modos y comparten proceso con las
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

MODES = ("same_session", "self_only")
# (filas, sesiones): una sesión de 64 o 256 activos y cuatro sesiones de 256.
SHAPES = ((64, 1), (256, 1), (1024, 4))
HIDDEN, HEADS, SEED, WARMUP, REPETITIONS = 64, 4, 42, 3, 20
PASSES = ("forward", "backward")


def protocol(device):
    return dict(
        kind="cross_asset_attention_cost_v1",
        device=device,
        modes=list(MODES),
        orders=[list(MODES), list(reversed(MODES))],
        shapes=[dict(rows=rows, sessions=sessions) for rows, sessions in SHAPES],
        hidden_size=HIDDEN,
        heads=HEADS,
        dtype="float32",
        sessions="rows split evenly by market US/CN and one decision instant per session",
        passes=list(PASSES),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La medida de coste no admite pasos de optimizador")


def session_keys(rows, sessions):
    import numpy as np

    size = rows // sessions
    markets = [("US", "CN")[(row // size) % 2] for row in range(rows)]
    instants = np.array(
        [
            np.datetime64("2015-01-02T21:00") + np.timedelta64(row // size, "D")
            for row in range(rows)
        ]
    )
    return markets, instants


def measure(module, encoded, keys, mode, synchronize):
    import torch

    def call():
        if mode == "forward":
            with torch.no_grad():
                module(encoded, *keys)
        else:
            module.zero_grad(set_to_none=True)
            module(encoded, *keys).square().sum().backward()
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

    from mars_titan.models.titans.cross_asset_attention import (
        CrossAssetAttention,
        CrossAssetAttentionConfig,
    )

    if device.startswith("cuda") and (device != "cuda:0" or not torch.cuda.is_available()):
        raise RuntimeError("La medida CUDA requiere cuda:0 disponible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    synchronize = torch.cuda.synchronize if device.startswith("cuda") else lambda: None
    guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
    modules = {
        mode: CrossAssetAttention(
            HIDDEN, CrossAssetAttentionConfig(heads=HEADS, mode=mode), seed=SEED
        ).to(device)
        for mode in MODES
    }
    # W_o empieza en cero. Se fija igual en ambos modos para que el gradiente recorra Q, K y V.
    generator = torch.Generator().manual_seed(SEED)
    weight = torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
    for module in modules.values():
        with torch.no_grad():
            module.out.weight.copy_(weight.to(device))
    inputs = {
        shape: (
            torch.randn(shape[0], HIDDEN, generator=generator).to(device),
            session_keys(*shape),
        )
        for shape in SHAPES
    }
    records = []
    try:
        for order in (MODES, tuple(reversed(MODES))):
            for shape in SHAPES:
                encoded, keys = inputs[shape]
                for kind in PASSES:
                    for mode in order:
                        samples = measure(modules[mode], encoded, keys, kind, synchronize)
                        records.append(
                            dict(
                                mode=mode, rows=shape[0], sessions=shape[1], pass_=kind, ms=samples
                            )
                        )
    finally:
        guard.remove()
    summary = []
    for mode in MODES:
        for rows, sessions in SHAPES:
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
                        sessions=sessions,
                        pass_=kind,
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
            mode: sum(value.numel() for value in modules[mode].parameters()) for mode in MODES
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
    result = run(arguments.device)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
