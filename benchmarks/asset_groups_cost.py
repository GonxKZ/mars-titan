"""Coste de los grupos de activos de H4 y de sus dos controles, sin aprendizaje (#783).

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo declarado
la llamada a `AssetGroupMembership` con representaciones aleatorias [R, H] en FP32 y ventanas
sintéticas con un factor de mercado, cuatro sectores y ruido propio, que solo fijan las formas.
Los bloques reparten las filas en sesiones de igual tamaño, con dos mercados y el mismo
instante, como un bloque de decisión completo.

`memberships` mide solo las pertenencias de cada sesión, que no dependen de los parámetros.
`forward` es la inferencia sin gradiente y `backward` añade la propagación de una suma de
cuadrados hasta los parámetros de H4, sin optimizador. Un gancho de `torch.optim` rechaza
cualquier paso. Las dos pasadas invierten el orden de los modos y comparten proceso con las
aplicaciones que ya existan, así que los percentiles son descriptivos. La salida debe ser nueva.
"""

import argparse
import json
import platform
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

MODES = ("past_window", "shuffled_rows", "current_sector")
# (filas, sesiones): una sesión de 64 o 256 activos y cuatro sesiones de 256.
SHAPES = ((64, 1), (256, 1), (1024, 4))
HIDDEN, SEED, WARMUP, REPETITIONS = 64, 42, 3, 20
PASSES = ("memberships", "forward", "backward")
SECTORS = 11


def protocol(device):
    return dict(
        kind="asset_groups_cost_v1",
        device=device,
        modes=list(MODES),
        orders=[list(MODES), list(reversed(MODES))],
        shapes=[dict(rows=rows, sessions=sessions) for rows, sessions in SHAPES],
        hidden_size=HIDDEN,
        configuration=dict(buckets=5, clusters=8, min_observations=36, iterations=32),
        dtype="float32",
        sessions="rows split evenly by market US/CN and one decision instant per session",
        windows="synthetic market, four sector factors and own noise, 64 sessions with presence",
        sectors=f"{SECTORS} synthetic sectors per market for the current sector control",
        passes=list(PASSES),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La medida de coste no admite pasos de optimizador")


def session_keys(rows, sessions):
    size = rows // sessions
    markets = [("US", "CN")[(row // size) % 2] for row in range(rows)]
    instants = [row // size for row in range(rows)]
    flows = [f"{market}/A{row:04}" for row, market in enumerate(markets)]
    return flows, markets, instants


def windows(rows, generator):
    """Ventanas [R, 64, 6] con cierres de un modelo de factores y un rango positivo."""
    import torch

    market = torch.randn(64, generator=generator) * 0.01
    factors = torch.randn(4, 64, generator=generator) * 0.01
    betas = 0.5 + torch.rand(rows, generator=generator)
    noise = torch.randn(rows, 64, generator=generator) * 0.004
    returns = betas[:, None] * market + factors[torch.arange(rows) % 4] + noise
    returns[:, 0] = 0
    close = returns.cumsum(1)
    close = close - close[:, :1]
    half = 0.001 + torch.rand(rows, 64, generator=generator) * 0.01
    opened = torch.cat([close[:, :1], close[:, :-1]], dim=1)
    present = torch.ones(rows, 64)
    return torch.stack(
        [opened, close + half, close - half, close, torch.zeros(rows, 64), present], dim=-1
    )


def sector_map():
    """Reparto sintético de todos los flujos de las formas medidas en sectores de su mercado."""
    from mars_titan.data.sector_map import SectorMap

    groups = sorted(
        {f"{market}/S{index:02}" for market in ("US", "CN") for index in range(SECTORS)}
        | {"US/unknown", "CN/unknown"}
    )
    index = {name: position for position, name in enumerate(groups)}
    sectors = {}
    for shape in SHAPES:
        flows, markets, _ = session_keys(*shape)
        for row, (flow, market) in enumerate(zip(flows, markets, strict=True)):
            sectors[flow] = index[f"{market}/S{row % SECTORS:02}"]
    assignments = tuple(sorted(sectors.items()))
    return SectorMap(tuple(groups), assignments, (("synthetic-sectors.csv", "0" * 64),))


def measure(module, inputs, kind, synchronize):
    import torch

    from mars_titan.data.asset_groups import session_memberships

    encoded, prices, flows, markets, instants = inputs
    sessions = module.sessions(markets, instants)
    config = module.config

    def call():
        if kind == "memberships":
            for key, members in sessions:
                index = torch.tensor(members, device=prices.device)
                if config.mode == "current_sector":
                    module.memberships(
                        prices.index_select(0, index), [flows[i] for i in members], key
                    )
                else:
                    session_memberships(
                        prices.index_select(0, index),
                        buckets=config.buckets,
                        clusters=config.clusters,
                        min_observations=config.min_observations,
                        iterations=config.iterations,
                    )
        elif kind == "forward":
            with torch.no_grad():
                module(*inputs)
        else:
            module.zero_grad(set_to_none=True)
            module(*inputs).square().sum().backward()
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

    from mars_titan.models.titans.asset_groups import AssetGroupMembership, AssetGroupsConfig

    if device.startswith("cuda") and (device != "cuda:0" or not torch.cuda.is_available()):
        raise RuntimeError("La medida CUDA requiere cuda:0 disponible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    synchronize = torch.cuda.synchronize if device.startswith("cuda") else lambda: None
    guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
    configs = {
        "past_window": AssetGroupsConfig(),
        "shuffled_rows": AssetGroupsConfig(mode="shuffled_rows"),
        "current_sector": AssetGroupsConfig(
            mode="current_sector",
            buckets=None,
            clusters=None,
            min_observations=None,
            iterations=None,
        ),
    }
    modules = {
        mode: AssetGroupMembership(
            HIDDEN,
            configs[mode],
            seed=SEED,
            sector_map=sector_map() if mode == "current_sector" else None,
        ).to(device)
        for mode in MODES
    }
    # W_o empieza en cero. Se fija igual en los tres modos para que el gradiente recorra H4.
    generator = torch.Generator().manual_seed(SEED)
    weight = torch.randn(HIDDEN, HIDDEN, generator=generator) / HIDDEN**0.5
    for module in modules.values():
        with torch.no_grad():
            module.out.weight.copy_(weight.to(device))
    inputs = {
        shape: (
            torch.randn(shape[0], HIDDEN, generator=generator).to(device),
            windows(shape[0], generator).to(device),
            *session_keys(*shape),
        )
        for shape in SHAPES
    }
    records = []
    try:
        for order in (MODES, tuple(reversed(MODES))):
            for shape in SHAPES:
                for kind in PASSES:
                    for mode in order:
                        samples = measure(modules[mode], inputs[shape], kind, synchronize)
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
