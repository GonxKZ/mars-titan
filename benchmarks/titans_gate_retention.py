"""Medir la retención de la memoria rápida de MAC con puertas v1 y con bias declarado.

Solo ejecuta pasos hacia delante, actualizaciones asociativas internas y derivadas del
objetivo exterior con `autograd.grad`. No crea optimizadores ni modifica parámetros.
"""

import argparse
import math
import platform
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from mars_titan.data.storage import atomic_json
from mars_titan.models.titans import GateBias, MACConfig, MemoryConfig, TitansMAC

CHOSEN = GateBias(alpha_half_life=256, eta=0.5, theta=0.05)
CHOSEN_ALPHA = -math.expm1(-math.log(2) / CHOSEN.alpha_half_life)
# (bias, alpha_max). La alternativa B no forma parte del núcleo y se emula aquí.
CANDIDATES = {
    "v1": (None, None),
    "A_h64_eta0.5": (GateBias(alpha_half_life=64, eta=0.5, theta=0.05), None),
    "A_h256_eta0.5": (CHOSEN, None),
    "A_h256_eta0.9": (GateBias(alpha_half_life=256, eta=0.9, theta=0.05), None),
    "A_h1024_eta0.5": (GateBias(alpha_half_life=1024, eta=0.5, theta=0.05), None),
    "B_alpha_max_2alpha256": (None, 2 * CHOSEN_ALPHA),
}
STREAMS = ("fused_norm_1", "unit_variance", "recurring_unit")
GRADIENT_ENDS = (64, 256, 1024, 4096)


class ScaledAlpha(nn.Module):
    """Alternativa B: α = α_max·σ(A x), devuelta como logit para la sigmoide del núcleo."""

    def __init__(self, linear, alpha_max):
        super().__init__()
        self.linear, self.alpha_max = linear, alpha_max

    def forward(self, values):
        alpha = self.alpha_max * self.linear(values).sigmoid()
        return torch.log(alpha) - torch.log1p(-alpha)


def build(candidate, *, dim, flows, dtype, device):
    gate_bias, alpha_max = candidate
    memory = MemoryConfig(
        dim=dim, depth=2, max_batch=flows, max_tokens=1, parameter_seed=42, gate_bias=gate_bias
    )
    mac = TitansMAC(
        MACConfig(memory=memory, heads=4, persistent_tokens=4, max_segment=1, parameter_seed=42),
        dtype=dtype,
        device=device,
    )
    if alpha_max is not None:
        mac.memory.alpha_projection = ScaledAlpha(mac.memory.alpha_projection, alpha_max)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(44)
        head = nn.Linear(dim, 1, dtype=dtype)
    return mac, head.to(device)


def tokens_for(stream, *, dim, flows, length, dtype, device):
    """El token fusionado del adaptador financiero inicial mide en torno a 1."""
    generator = torch.Generator().manual_seed(2026)

    def noise():
        return torch.randn(flows, 1, dim, generator=generator, dtype=dtype)

    if stream == "fused_norm_1":
        tokens = [noise() * dim**-0.5 for _ in range(length)]
    elif stream == "unit_variance":
        tokens = [noise() for _ in range(length)]
    else:
        prototypes = [noise() for _ in range(4)]
        tokens = [prototypes[step % 4] + 0.3 * noise() for step in range(length)]
    return [token.to(device) for token in tokens]


def summary(values):
    tensor = torch.tensor(values, dtype=torch.float64)
    return {"min": tensor.min().item(), "mean": tensor.mean().item(), "max": tensor.max().item()}


class RateProbe:
    """Registrar tasas y la asociación media E[v kᵀ] fuera de las ventanas con grafo."""

    def __init__(self, memory, flows, dim, device):
        self.enabled, self.memory = True, memory
        self.values = {"alpha": [], "eta": [], "theta": [], "value_norm": []}
        self.association = torch.zeros(flows, dim, dim, dtype=torch.float64, device=device)
        self.count, self.keys = 0, None
        self.hooks = [
            memory.alpha_projection.register_forward_hook(self._rate("alpha", 1.0)),
            memory.eta_projection.register_forward_hook(self._rate("eta", 1.0)),
            memory.theta_projection.register_forward_hook(
                self._rate("theta", memory.config.theta_max)
            ),
            memory.key_projection.register_forward_hook(self._keys),
            memory.value_projection.register_forward_hook(self._values),
        ]

    def _rate(self, name, scale):
        def hook(module, args, output):
            if self.enabled:
                self.values[name].append(scale * output.detach().sigmoid().mean().item())

        return hook

    def _keys(self, module, args, output):
        if self.enabled:
            self.keys = F.normalize(output.detach().double(), dim=-1, eps=1e-12)

    def _values(self, module, args, output):
        if self.enabled:
            values = output.detach().double()
            self.values["value_norm"].append(values.norm(dim=-1).mean().item())
            self.association += values.unsqueeze(-1) * self.keys.unsqueeze(-2)
            self.count += 1

    def close(self):
        for hook in self.hooks:
            hook.remove()


def outer_gradients(mac, head, state, tokens, probe):
    """Derivar un funcional lineal de las salidas, equivalente en escala a la derivada del MAE."""
    probe.enabled = False
    total = 0
    for token in tokens:
        output, state = mac(token, state, differentiable=True)
        total = total + head(output).sum()
    probe.enabled = True
    named = {"head.weight": head.weight}
    named.update({f"mac.{name}": value for name, value in mac.named_parameters()})
    gradients = torch.autograd.grad(total, tuple(named.values()), allow_unused=True)
    return {
        name: 0.0 if gradient is None else gradient.norm().item()
        for name, gradient in zip(named, gradients, strict=True)
    }


def trajectory(candidate, stream, *, dim, flows, length, window, dtype, device):
    mac, head = build(candidate, dim=dim, flows=flows, dtype=dtype, device=device)
    probe = RateProbe(mac.memory, flows, dim, device)
    tokens = tokens_for(stream, dim=dim, flows=flows, length=length, dtype=dtype, device=device)
    checkpoints = sorted({0, 1, 8, 64, 256, 512, 1024, 2048, length} & set(range(length + 1)))
    state = mac.initial_state(flows)
    norms = {0: [w.norm(dim=(1, 2)).mean().item() for w in state.memory.weights]}
    output_norms, layer_norms, gradients = {}, [], {}
    for step in range(1, length + 1):
        end = step + window - 1
        if end in GRADIENT_ENDS and end <= length:
            gradients[end] = outer_gradients(mac, head, state, tokens[step - 1 : end], probe)
        with torch.no_grad():
            output, state = mac(tokens[step - 1], state)
        current = [w.norm(dim=(1, 2)).mean().item() for w in state.memory.weights]
        if not all(math.isfinite(value) for value in current):
            raise ValueError("La norma de la memoria dejó de ser finita")
        layer_norms.append(current)
        if step in checkpoints:
            norms[step] = current
            output_norms[step] = output.norm(dim=-1).mean().item()
    probe.close()
    alpha, eta, theta = (summary(probe.values[name]) for name in ("alpha", "eta", "theta"))
    value = summary(probe.values["value_norm"])
    association = torch.linalg.matrix_norm(probe.association / probe.count, ord=2).mean().item()
    forget = (1 - eta["mean"]) * alpha["mean"]
    return {
        "weight_norm_by_layer": {str(step): values for step, values in norms.items()},
        "weight_norm_range_after_64": [
            summary([values[layer] for values in layer_norms[63:]]) for layer in range(2)
        ],
        "output_norm": {str(step): value for step, value in output_norms.items()},
        "rates": {"alpha": alpha, "eta": eta, "theta": theta},
        "value_norm": value,
        "mean_association_spectral_norm": association,
        "write_forget_ratio_per_token": theta["mean"] * value["mean"] / forget,
        "write_forget_ratio_mean_association": theta["mean"] * association / forget,
        "outer_gradient_norms": {str(end): values for end, values in gradients.items()},
    }


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
    for dtype in (torch.float64, torch.float32):
        for stream in STREAMS:
            for name, candidate in CANDIDATES.items():
                if dtype == torch.float32 and name not in ("v1", "A_h256_eta0.5"):
                    continue
                key = f"{str(dtype).removeprefix('torch.')}/{stream}/{name}"
                results[key] = trajectory(
                    candidate,
                    stream,
                    dim=args.dim,
                    flows=args.flows,
                    length=args.length,
                    window=args.window,
                    dtype=dtype,
                    device=device,
                )
                print(key, results[key]["weight_norm_by_layer"][str(args.length)], flush=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    receipt = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(UTC).isoformat(),
        "commit": commit,
        "command": "uv run --no-sync python benchmarks/titans_gate_retention.py --output <recibo>",
        "environment": "CUDA_VISIBLE_DEVICES=-1, OMP_NUM_THREADS=2, MKL_NUM_THREADS=2",
        "scope": "Dinámica técnica con fixtures aleatorios en CPU, sin optimizador ni datos",
        "hardware": {
            "machine": platform.machine(),
            "processor": platform.processor(),
            "device": str(device),
            "cuda_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cuda_peak_bytes": torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else None,
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
        "streams": {
            "fused_norm_1": "N(0, I/D), norma cercana a la del token fusionado inicial",
            "unit_variance": "N(0, I)",
            "recurring_unit": "cuatro prototipos N(0, I) alternos más 0,3·N(0, I)",
        },
        "outer_objective": "suma de head(salida) en los últimos `window` pasos, con grafo",
        "window": args.window,
        "half_life_formula": "log(0,5)/log(1 − α)",
        "half_life_examples": {
            str(alpha): math.log(0.5) / math.log1p(-alpha)
            for alpha in (0.5, 0.1, 0.0107717, CHOSEN_ALPHA)
        },
        "candidates": {
            name: {
                "gate_bias": None if bias is None else vars(bias),
                "alpha_max": alpha_max,
            }
            for name, (bias, alpha_max) in CANDIDATES.items()
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
