"""Coste de las configuraciones de D1 con cuatro pasos del puente, sin aprendizaje (#751).

Sin --run imprime el protocolo y no importa Torch. Con --run mide en el dispositivo
declarado el lector episódico con puente recurrente sobre un banco ficticio en FP32. Compara
K=2 con J=2, K=4 con J=1 y K=1 con J=4. Cada configuración se mide con la pérdida final y,
cuando tiene más de un refinamiento, con la pérdida en las profundidades 2 y 4.

Cada llamada calcula lectura, cabeza, pérdida y propagación hacia atrás hasta los
parámetros del lector, sin optimizador, y un gancho de `torch.optim` rechaza cualquier
paso. Las dos pasadas invierten el orden de las configuraciones y comparten proceso con las
aplicaciones que ya existan, así que los percentiles son descriptivos. La salida debe ser
nueva.
"""

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.storage import atomic_json

ARMS = (
    ("k2_j2_final", 2, 2, None),
    ("k2_j2_per_step", 2, 2, (1, 2)),
    ("k4_j1_final", 4, 1, None),
    ("k4_j1_per_step", 4, 1, (2, 4)),
    ("k1_j4_final", 1, 4, None),
)
ROWS = (64, 256)
HIDDEN, EPISODES, SEED, WARMUP, REPETITIONS = 64, 256, 42, 3, 20
CODEC, CONTEXT = "a" * 64, "b" * 64
# El lector y su puente comparten el presupuesto de trabajo. Es solo la cota de la
# comprobación previa con 256 filas y K=4, no cambia el cálculo ni la memoria usada.
WORKING_BYTES = 512 * 1024**2


def protocol(device):
    return dict(
        kind="nested_equal_depth_readout_cost_v1",
        device=device,
        arms=[dict(name=n, refinements=k, steps=j, supervised=s) for n, k, j, s in ARMS],
        orders=[[arm[0] for arm in ARMS], [arm[0] for arm in reversed(ARMS)]],
        rows=list(ROWS),
        hidden_size=HIDDEN,
        episodes=EPISODES,
        dtype="float32",
        bridge="three views, ring, cycles=0, recurrence budget 1216",
        head="nn.Linear(hidden, 1) with l1 loss",
        call="readout forward, head, loss and backward to the readout parameters",
        warmup=WARMUP,
        repetitions=REPETITIONS,
        optimizer_steps="forbidden by a torch.optim pre-hook",
    )


def forbid_optimizer_steps(optimizer, args, kwargs):
    raise RuntimeError("La medida de coste no admite pasos de optimizador")


def build(refinements, steps, device):
    import torch

    from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
    from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

    bridge = EvidenceBridgeConfig(
        codec_id=CODEC,
        hidden_size=HIDDEN,
        cycles=0,
        recurrence=BridgeRecurrenceConfig(steps, 1216),
        max_working_bytes=WORKING_BYTES,
    )
    config = EpisodicReadoutConfig(
        CODEC,
        hidden_size=HIDDEN,
        refinements=refinements,
        seed=SEED,
        max_working_bytes=WORKING_BYTES,
        evidence_bridge=bridge,
    )
    return EpisodicReadout(config, dtype=torch.float32, device=device)


def bank(device):
    import torch

    from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

    generator = torch.Generator().manual_seed(SEED)
    # El banco guarda claves de norma uno.
    keys = torch.nn.functional.normalize(torch.randn(EPISODES, 64, generator=generator), dim=-1)
    return EpisodeSnapshot.create(
        keys=keys,
        values=torch.randn(EPISODES, 64, generator=generator),
        labels=torch.randn(EPISODES, generator=generator, dtype=torch.float64) / 100,
        ids=torch.arange(1, EPISODES + 1, dtype=torch.int64),
        decision_at=torch.ones(EPISODES, dtype=torch.int64),
        available_at=torch.ones(EPISODES, dtype=torch.int64),
        maturity_at=torch.full((EPISODES,), 2, dtype=torch.int64),
        cutoff=10,
        codec_id=CODEC,
        context_id=CONTEXT,
        device=device,
        dtype=torch.float32,
    )


def measure(model, head, snapshot, state, target, supervised, synchronize):
    import torch

    from mars_titan.models.titans.episodic_readout import EpisodicPrediction, head_predictions
    from mars_titan.training.mars_titan_run import readout_loss

    index = torch.arange(len(state), device=state.device)

    def call():
        model.zero_grad(set_to_none=True)
        result = model(state, snapshot, context_id=CONTEXT, cutoff=10, differentiable=True)
        point, quantiles = head_predictions(head, result.state)
        prediction = EpisodicPrediction(point, result, quantiles)
        loss = readout_loss(prediction, head, index, target, loss="l1", supervised=supervised)
        loss.backward()
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

    if device.startswith("cuda") and (device != "cuda:0" or not torch.cuda.is_available()):
        raise RuntimeError("La medida CUDA requiere cuda:0 disponible")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    synchronize = torch.cuda.synchronize if device.startswith("cuda") else lambda: None
    load = os.getloadavg()
    guard = register_optimizer_step_pre_hook(forbid_optimizer_steps)
    models = {name: build(k, j, device) for name, k, j, _ in ARMS}
    generator = torch.Generator().manual_seed(SEED + 1)
    head = torch.nn.Linear(HIDDEN, 1).to(device)
    snapshot = bank(device)
    states = {rows: torch.randn(rows, HIDDEN, generator=generator).to(device) for rows in ROWS}
    targets = {rows: torch.randn(rows, generator=generator).to(device) / 100 for rows in ROWS}
    records = []
    try:
        for order in (ARMS, tuple(reversed(ARMS))):
            for rows in ROWS:
                for name, _, _, supervised in order:
                    samples = measure(
                        models[name],
                        head,
                        snapshot,
                        states[rows],
                        targets[rows],
                        supervised,
                        synchronize,
                    )
                    records.append(dict(arm=name, rows=rows, ms=samples))
    finally:
        guard.remove()
    summary = []
    for name, *_ in ARMS:
        for rows in ROWS:
            pooled = sorted(
                value
                for record in records
                if (record["arm"], record["rows"]) == (name, rows)
                for value in record["ms"]
            )
            summary.append(
                dict(
                    arm=name,
                    rows=rows,
                    p50_ms=statistics.median(pooled),
                    p95_ms=pooled[min(len(pooled) - 1, round(0.95 * (len(pooled) - 1)))],
                )
            )
    return dict(
        protocol=protocol(device),
        finished_at=datetime.now(UTC).isoformat(),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        environment=dict(
            load_average_at_start=load,
            load_average_at_end=os.getloadavg(),
            python=sys.version.split()[0],
            torch=torch.__version__,
            machine=platform.machine(),
            threads=torch.get_num_threads(),
            gpu=torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        ),
        parameters={
            name: sum(value.numel() for value in model.parameters())
            for name, model in models.items()
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
