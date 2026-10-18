"""Coste del lector y de sus rutas, con datos técnicos y referencia numérica."""

import argparse
import importlib
import json
import os
import platform
import resource
import time
from contextlib import contextmanager, nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.quantile_head import QuantileHead
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.refinement_policy import FEATURES, RoutePolicy, apply_routes
from mars_titan.models.titans.refinement_routes import run_route

CODEC, CONTEXT = "a" * 64, "b" * 64


def source_identity():
    names = (
        "mars_titan.models.titans.episodic_readout",
        "mars_titan.models.titans.episodic_snapshot",
        "mars_titan.models.titans.refinement_policy",
        "mars_titan.models.titans.refinement_routes",
        "mars_titan.models.titans.config",
        "mars_titan.models.titans.state",
        "mars_titan.models.quantile_head",
        "mars_titan.training.experiment_resources",
    )
    return dict(
        benchmark=sha256(Path(__file__)),
        **{name: sha256(Path(importlib.import_module(name).__file__)) for name in names},
    )


@contextmanager
def strict_numerics():
    precision = torch.get_float32_matmul_precision()
    tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        yield
    finally:
        torch.set_float32_matmul_precision(precision)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32


def fixture(batch, episodes, device):
    """Pesos y entradas técnicos con RNG propio. No representan resultados financieros."""
    generator = torch.Generator().manual_seed(20261018)
    config = EpisodicReadoutConfig(
        CODEC,
        hidden_size=64,
        refinements=4,
        episode_selection="first_read",
        max_working_bytes=128 * 1024**2,
    )
    reader = EpisodicReadout(config, device=device).eval().requires_grad_(False)
    keys = torch.randn((episodes, 64), generator=generator)
    keys = keys / torch.linalg.vector_norm(keys.double(), dim=-1, keepdim=True).float()
    snapshot = EpisodeSnapshot.create(
        keys=keys,
        values=torch.randn((episodes, 64), generator=generator),
        labels=torch.randn(episodes, generator=generator, dtype=torch.float64) * 0.01,
        ids=torch.arange(1, episodes + 1, dtype=torch.int64),
        decision_at=torch.ones(episodes, dtype=torch.int64),
        available_at=torch.ones(episodes, dtype=torch.int64),
        maturity_at=torch.full((episodes,), 2, dtype=torch.int64),
        cutoff=10,
        codec_id=CODEC,
        context_id=CONTEXT,
        device=device,
        dtype=torch.float32,
    )
    base = torch.randn((batch, 64), generator=generator).to(device)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(20261018)
        head = QuantileHead(64).eval().requires_grad_(False).to(device)
    return reader, snapshot, base, head


def calls(reader, snapshot, base, head):
    options = dict(context_id=CONTEXT, cutoff=10)
    first = EpisodicReadout(replace(reader.config, refinements=1), device=base.device)
    with torch.no_grad():
        for (_, target), (_, source) in zip(
            first.named_parameters(), reader.named_parameters(), strict=True
        ):
            target.copy_(source)
    first.eval().requires_grad_(False)
    width = len(FEATURES)
    policy = RoutePolicy(
        mean=torch.zeros(width, device=base.device),
        scale=torch.ones(width, device=base.device),
        coefficients=torch.zeros((width + 1, 5), device=base.device),
        support_min=torch.full((width,), -100.0, device=base.device),
        support_max=torch.full((width,), 100.0, device=base.device),
        costs=torch.zeros(5, device=base.device),
        min_gain=0.0,
    )
    mixed = torch.arange(len(base), device=base.device) % 6

    def first_with_head():
        state = first(base, snapshot, **options).state
        head(state)
        return state

    return {
        "legacy_k1": lambda: first(base, snapshot, **options).state,
        "legacy_k1_with_head": first_with_head,
        "legacy_first_read_k4": lambda: reader(base, snapshot, **options).state,
        "cached_k4cc": lambda: run_route(reader, base, snapshot, route="k4cc", **options).state,
        "cached_k4cr": lambda: run_route(reader, base, snapshot, route="k4cr", **options).state,
        "policy_k1": lambda: (
            apply_routes(reader, base, head, snapshot, policy=policy, **options).state
        ),
        "mixed_explicit_routes": lambda: (
            apply_routes(reader, base, head, snapshot, choices=mixed, **options).state
        ),
    }


def _statistics(seconds, rows):
    values = np.asarray(seconds)
    return dict(
        repetitions=len(values),
        seconds=values.tolist(),
        p50_ms=float(np.quantile(values, 0.50) * 1000),
        p95_ms=float(np.quantile(values, 0.95) * 1000),
        p99_ms=float(np.quantile(values, 0.99) * 1000),
        mean_ms=float(values.mean() * 1000),
        rows_per_second=float(rows / values.mean()),
    )


def _case(batch, episodes, device, warmup, repeats):
    reader, snapshot, base, head = fixture(batch, episodes, device)
    functions = calls(reader, snapshot, base, head)
    synchronize = (lambda: torch.cuda.synchronize(0)) if device == "cuda:0" else lambda: None
    with torch.no_grad():
        reference = functions["legacy_first_read_k4"]()
        cached = functions["cached_k4cc"]()
        if not torch.equal(reference, cached):
            raise ValueError("La caché no conserva los bits del lector de referencia")
        first = functions["legacy_k1"]()
        if not torch.equal(first, functions["policy_k1"]()):
            raise ValueError("La puerta K1 no conserva su referencia")
        expected = {name: function() for name, function in functions.items()}
        for function in functions.values():
            for _ in range(warmup):
                function()
        synchronize()
        timings = {name: [] for name in functions}
        peaks = {name: 0 for name in functions}
        names = list(functions)
        for repetition in range(repeats):
            order = names[repetition % len(names) :] + names[: repetition % len(names)]
            for name in order:
                synchronize()
                if device == "cuda:0":
                    torch.cuda.reset_peak_memory_stats(0)
                began = time.perf_counter()
                result = functions[name]()
                synchronize()
                timings[name].append(time.perf_counter() - began)
                if device == "cuda:0":
                    peaks[name] = max(peaks[name], torch.cuda.max_memory_allocated(0))
                if not torch.isfinite(result).all():
                    raise ValueError("El benchmark produjo un estado no finito")
                if not torch.equal(result, expected[name]):
                    raise ValueError("Las repeticiones no conservan los mismos bits")
        return dict(
            batch=batch,
            episodes=episodes,
            hidden_size=64,
            neighbors=8,
            dtype="torch.float32",
            parity="bitwise_cached_k4cc_and_legacy_first_read_k4",
            variants={
                name: dict(
                    _statistics(values, batch),
                    peak_allocated_bytes=peaks[name] if device == "cuda:0" else None,
                )
                for name, values in timings.items()
            },
        )


def measure(
    *,
    device,
    batches=(1, 64, 256),
    episodes=(0, 128, 1024),
    warmup=5,
    repeats=30,
    concurrent_workloads=(),
):
    if (
        device not in ("cpu", "cuda:0")
        or type(warmup) is not int
        or not 0 <= warmup <= 50
        or type(repeats) is not int
        or not 2 <= repeats <= 200
    ):
        raise ValueError("Dispositivo, calentamiento o repeticiones fuera del contrato")
    if not batches or not episodes or len(batches) > 8 or len(episodes) > 8:
        raise ValueError("La parrilla de tamaños debe ser no vacía y acotada")
    if any(type(n) is not int or not 1 <= n <= 256 for n in batches) or any(
        type(n) is not int or not 0 <= n <= 1024 for n in episodes
    ):
        raise ValueError("El lote y el banco exceden el contrato del lector")
    lease = nullcontext()
    if device == "cuda:0":
        from mars_titan.training.experiment_resources import GpuLease

        lease = GpuLease()
    started = time.perf_counter()
    code = source_identity()
    with lease as admitted, strict_numerics():
        if device == "cuda:0" and not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. No se cambia a CPU")
        cases = [
            _case(batch, count, device, warmup, repeats) for batch in batches for count in episodes
        ]
        if device == "cuda:0":
            admitted.check()
        gpu = admitted.record if device == "cuda:0" else None
    if source_identity() != code:
        raise ValueError("El código ha cambiado durante la medida")
    return dict(
        schema_version=1,
        kind="refinement_component_cost",
        measured_at_utc=datetime.now(UTC).isoformat(),
        scope="reader_only_without_mac_data_loading_or_training",
        device=device,
        gpu=gpu,
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        numpy=np.__version__,
        threads=torch.get_num_threads(),
        concurrent_workloads=list(concurrent_workloads),
        host_load_average=list(os.getloadavg()),
        warmup=warmup,
        repeats=repeats,
        precision="fp32_no_tf32",
        workload="seeded_technical_fixture_not_financial_quality",
        workload_seed=20261018,
        source_sha256=code,
        optimizer_steps=0,
        cases=cases,
        wall_seconds=time.perf_counter() - started,
        process_peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        limitations=[
            "Los percentiles son descriptivos del número de repeticiones.",
            "Solo k4cc se compara por paridad con first_read.",
            "policy_k1 incluye cabeza y puerta. Las rutas fijas medidas devuelven estado.",
            "El pico de RAM corresponde al proceso completo.",
            "No se han medido energía, coste monetario ni calidad predictiva.",
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256])
    parser.add_argument("--episodes", type=int, nargs="+", default=[0, 128, 1024])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrent-workload", action="append", default=[])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("La medida necesita un destino nuevo")
    record = measure(
        device=args.device,
        batches=args.batches,
        episodes=args.episodes,
        warmup=args.warmup,
        repeats=args.repeats,
        concurrent_workloads=args.concurrent_workload,
    )
    atomic_json(args.output, record)
    print(
        json.dumps(
            dict(
                output=str(args.output),
                cases=len(record["cases"]),
                wall_seconds=record["wall_seconds"],
            )
        )
    )


if __name__ == "__main__":
    main()
