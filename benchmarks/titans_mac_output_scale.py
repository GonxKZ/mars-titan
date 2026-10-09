"""Medir la escala de la salida de MAC, `y ⊙ M(y)`, con la memoria v1 y con residual y LN.

Reutiliza los flujos de entrada y la derivada exterior de `titans_gate_retention.py`.
Solo ejecuta pasos hacia delante, escrituras asociativas internas y `autograd.grad` de un
funcional lineal de la salida. No crea optimizadores ni modifica parámetros.
"""

import argparse
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from titans_gate_retention import CHOSEN, GRADIENT_ENDS, STREAMS, RateProbe, outer_gradients
from titans_gate_retention import tokens_for as retention_tokens
from torch import nn

from mars_titan.data.storage import atomic_json
from mars_titan.models.titans import MACConfig, MemoryConfig, TitansMAC

# (gate_bias, residual_layer_norm)
CANDIDATES = {
    "gate_bias": (CHOSEN, False),
    "gate_bias_residual_layer_norm": (CHOSEN, True),
    "v1_residual_layer_norm": (None, True),
}
CHECKPOINTS = (1, 8, 64, 256, 1024, 2048, 4096)


def build(candidate, *, dim, flows, dtype, device):
    gate_bias, residual = candidate
    memory = MemoryConfig(
        dim=dim,
        depth=2,
        max_batch=flows,
        max_tokens=1,
        parameter_seed=42,
        gate_bias=gate_bias,
        residual_layer_norm=residual,
    )
    mac = TitansMAC(
        MACConfig(memory=memory, heads=4, persistent_tokens=4, max_segment=1, parameter_seed=42),
        dtype=dtype,
        device=device,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        head = nn.Linear(dim, 1, dtype=dtype)
    return mac, head.to(device)


def quantiles(values):
    flat = values.detach().abs().flatten().double()
    levels = torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64, device=flat.device)
    return dict(zip(("p10", "p50", "p90"), torch.quantile(flat, levels).tolist(), strict=True))


def trajectory(candidate, stream, *, dim, flows, length, window, dtype, device):
    mac, head = build(candidate, dim=dim, flows=flows, dtype=dtype, device=device)
    probe = RateProbe(mac.memory, flows, dim, device)
    captured = {}
    hook = mac.attention.register_forward_hook(
        lambda module, args, output: captured.update(y=output[0][:, -1].detach())
    )
    tokens = retention_tokens(
        stream, dim=dim, flows=flows, length=length, dtype=dtype, device=device
    )
    state = mac.initial_state(flows)
    scales, gradients = {}, {}
    for step in range(1, length + 1):
        end = step + window - 1
        if end in GRADIENT_ENDS and end <= length:
            gradients[end] = outer_gradients(mac, head, state, tokens[step - 1 : end], probe)
        with torch.no_grad():
            output, state = mac(tokens[step - 1], state)
            if step in CHECKPOINTS or step == length:
                y = captured["y"]
                read = mac.memory.read(y.unsqueeze(1), state.memory).squeeze(1)
                scales[str(step)] = dict(
                    y_norm=y.norm(dim=-1).mean().item(),
                    memory_read_norm=read.norm(dim=-1).mean().item(),
                    output_norm=output.norm(dim=-1).mean().item(),
                    output_abs=quantiles(output),
                    weight_norm_by_layer=[
                        w.norm(dim=(1, 2)).mean().item() for w in state.memory.weights
                    ],
                )
    hook.remove()
    probe.close()
    keep = ("head.weight", "mac.memory.alpha_projection.weight", "mac.query_projection.weight")
    return dict(
        scales=scales,
        outer_gradient_norms={
            str(end): {name: values[name] for name in keep} for end, values in gradients.items()
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--flows", type=int, default=4)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda:0"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se solicitó cuda:0 y no hay dispositivo CUDA visible")
    started = time.perf_counter()
    results = {}
    for stream in STREAMS:
        for name, candidate in CANDIDATES.items():
            key = f"float64/{stream}/{name}"
            results[key] = trajectory(
                candidate,
                stream,
                dim=args.dim,
                flows=args.flows,
                length=args.length,
                window=args.window,
                dtype=torch.float64,
                device=device,
            )
            print(key, results[key]["scales"][str(args.length)]["output_norm"], flush=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "command": (
            "uv run --no-sync python benchmarks/titans_mac_output_scale.py --output <recibo>"
        ),
        "environment": "CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2",
        "scope": "Escala técnica con fixtures aleatorios en CPU, sin optimizador ni datos",
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": str(device),
        },
        "versions": {"python": platform.python_version(), "torch": torch.__version__},
        "threads": args.threads,
        "shapes": {
            "dim": args.dim,
            "depth": 2,
            "flows": args.flows,
            "segment": 1,
            "heads": 4,
            "persistent_tokens": 4,
            "length": args.length,
        },
        "streams": "Los tres flujos de benchmarks/titans_gate_retention.py",
        "measures": {
            "y_norm": "norma media de la salida de atención y",
            "memory_read_norm": "norma media de M(y) con la memoria posterior a la escritura",
            "output_norm": "norma media de y ⊙ M(y)",
            "output_abs": "cuantiles 10, 50 y 90 de |y ⊙ M(y)| por elemento",
        },
        "outer_objective": "suma de head(salida) en los últimos `window` pasos, con grafo",
        "window": args.window,
        "candidates": {
            name: {"gate_bias": None if bias is None else vars(bias), "residual_layer_norm": res}
            for name, (bias, res) in CANDIDATES.items()
        },
        "results": results,
        "optimizer_steps": 0,
        "training_runs": 0,
        "gpu_workloads": int(device.type == "cuda"),
        "seconds": time.perf_counter() - started,
    }
    atomic_json(args.output, receipt)


if __name__ == "__main__":
    main()
