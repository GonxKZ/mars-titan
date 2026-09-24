"""Comparar E/S de episodios y evaluación mínima en CPU, sin entrenar ni usar CUDA."""

import argparse
import cProfile
import importlib
import json
import os
import platform
import pstats
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa

from mars_titan.data.storage import atomic_json, sha256


class DiagnosticRows:
    """Entradas analíticas residentes para aislar el recorrido de evaluación."""

    def __init__(self, count):
        self.count = count
        self.manifest = {"counts": {"validation": count}}
        self.values = (np.arange(count * 16, dtype=np.float32) % 97 / 97).reshape(count, 16)
        self.target = (np.arange(count, dtype=np.float32) % 23) / 23
        self.markets = np.where(np.arange(count) % 2 == 0, "US", "CN")
        self.times = (
            1_546_300_800_000_000 + np.arange(count, dtype=np.int64) // 128 * 86_400_000_000
        )
        self.ids = [f"{market}/FIC{i % 128:06d}/{i}" for i, market in enumerate(self.markets)]

    def batches(self, *, batch_size, **_options):
        for start in range(0, self.count, batch_size):
            end = min(start + batch_size, self.count)
            yield dict(
                inputs={"values": self.values[start:end]},
                target=self.target[start:end],
                market=self.markets[start:end],
                prediction_at=self.times[start:end],
                sample_ids=self.ids[start:end],
            )


def episode_work(args):
    from episode_pipeline import pipeline

    return lambda: pipeline(args.work, 1)


def evaluation_work(args):
    import torch

    engine = importlib.import_module("mars_titan.training.reference_run")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    dataset = DiagnosticRows(1024 if args.size == "small" else 65536)
    model = torch.nn.Linear(16, 1)
    with torch.no_grad():
        model.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(1, 16) / 64)
        model.bias.fill_(0.125)

    class Predictor(torch.nn.Module):
        def forward(self, values):
            return model(values["values"]).squeeze(-1)

    predictor = Predictor()
    destination = args.work / f"predictions-{os.getpid()}.parquet" if args.persist else None

    def evaluate():
        started = time.perf_counter()
        # Adaptador exclusivo del diagnóstico. La ruta científica sigue exigiendo CUDA.
        with (
            patch.object(
                engine,
                "_inputs",
                lambda batch, _device: {"values": torch.from_numpy(batch["inputs"]["values"])},
            ),
            patch.object(torch.cuda, "synchronize", lambda _device: None),
        ):
            metrics = engine._evaluate(predictor, dataset, 512, destination=destination)
        elapsed = time.perf_counter() - started
        return dict(
            pipeline_seconds=elapsed,
            samples=dataset.count,
            samples_per_second=dataset.count / elapsed,
            input_bytes=dataset.values.nbytes + dataset.target.nbytes + dataset.times.nbytes,
            metrics={
                key: value
                for key, value in metrics.items()
                if key not in {"elapsed_seconds", "samples_per_second"}
            },
            prediction_sha256=sha256(destination) if destination else None,
            prediction_bytes=destination.stat().st_size if destination else 0,
            torch_version=torch.__version__,
        )

    return evaluate


def prepare(args):
    args.work.mkdir(parents=True, exist_ok=True)
    if args.kind == "evaluation" or (args.work / "episodes/manifest.json").exists():
        return
    from mars_titan.episodes.storage import write_world
    from mars_titan.episodes.worlds import WorldConfig, generate_world
    from mars_titan.simulation.market import MarketTape
    from mars_titan.simulation.storage import write_tape

    assets, sessions, context = (4, 40, 8) if args.size == "small" else (128, 256, 64)
    world = generate_world(
        WorldConfig(assets=assets, sessions=sessions, context=context, seed=4242)
    )
    write_world(world, args.work / "episodes")
    write_tape(
        MarketTape.from_world(world, lambda inputs: 0.002 * inputs["news"][:, 0]),
        args.work / "market",
    )


def worker(args):
    operation = episode_work(args) if args.kind == "episodes" else evaluation_work(args)
    operation()
    if args.profile:
        profiler = cProfile.Profile()
        result = profiler.runcall(operation)
        selected = []
        for (file, line, function), values in sorted(
            pstats.Stats(profiler).stats.items(), key=lambda entry: entry[1][3], reverse=True
        ):
            if (
                Path(file).name in {"storage.py", "reference_run.py", "session_metrics.py"}
                or "parquet/core.py" in file
            ):
                selected.append(
                    dict(
                        file=Path(file).name,
                        line=line,
                        function=function,
                        calls=values[1],
                        self_seconds=values[2],
                        cumulative_seconds=values[3],
                    )
                )
        result["profile"] = selected[:20]
    else:
        result = operation()
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    result["worker_peak_rss_bytes"] = int(status["VmHWM"].split()[0]) * 1024
    result.pop("process_lifetime_peak_rss_bytes", None)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("episodes", "evaluation"), required=True)
    parser.add_argument("--size", choices=("small", "representative"), required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--persist", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare_only:
        prepare(args)
        return
    if args.worker:
        worker(args)
        return
    if args.output is None or not 3 <= args.repetitions <= 10:
        raise ValueError("Indica una salida y entre tres y diez repeticiones")
    command = [
        sys.executable,
        __file__,
        "--worker",
        "--kind",
        args.kind,
        "--size",
        args.size,
        "--work",
        str(args.work),
    ]
    if args.persist:
        command.append("--persist")
    env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    # La preparación no debe elevar el máximo de RSS heredado por los procesos medidos.
    subprocess.run([*command, "--prepare-only"], env=env, check=True, timeout=120)
    runs = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        output = subprocess.check_output(command, env=env, text=True, timeout=120)
        result = json.loads(output)
        result["total_process_seconds"] = time.perf_counter() - started
        runs.append(result)
    parity_keys = ("input_sha256", "financial_validation", "metrics", "prediction_sha256")
    for key in parity_keys:
        if any(run.get(key) != runs[0].get(key) for run in runs):
            raise ValueError(f"Las repeticiones difieren en {key}")
    profiled = json.loads(
        subprocess.check_output([*command, "--profile"], env=env, text=True, timeout=120)
    )
    for key in parity_keys:
        if profiled.get(key) != runs[0].get(key):
            raise ValueError(f"El perfil difiere en {key}")
    summary = {}
    for name in ("pipeline_seconds", "total_process_seconds", "worker_peak_rss_bytes"):
        values = [run[name] for run in runs]
        summary[name] = dict(
            median=float(np.median(values)),
            min=min(values),
            max=max(values),
            std=float(np.std(values, ddof=1)),
        )
    cpu = subprocess.check_output(["rg", "-m", "1", "^model name", "/proc/cpuinfo"], text=True)
    report = dict(
        measured_at=datetime.now(UTC).isoformat(),
        domain="technical",
        device="cpu",
        kind=args.kind,
        size=args.size,
        persist=args.persist,
        final_test_opened=False,
        hardware=dict(cpu=cpu.strip().split(":", 1)[1].strip(), os=platform.platform()),
        software=dict(
            python=platform.python_version(), numpy=np.__version__, pyarrow=pa.__version__
        ),
        repetitions=args.repetitions,
        warmup_runs_per_process=1,
        numerical_threads=1,
        code_sha256={
            name: sha256(Path(name))
            for name in (
                "src/mars_titan/episodes/storage.py",
                "src/mars_titan/training/reference_run.py",
                "benchmarks/episode_pipeline.py",
                "benchmarks/evaluation_io.py",
            )
        },
        summary=summary,
        runs=runs,
        profile=profiled["profile"],
        limits=[
            "Datos sintéticos, caché del sistema caliente y otras cargas activas.",
            "Evaluación con modelo lineal mínimo en CPU. No mide entrenamiento ni aceleración GPU.",
            "El tiempo de proceso incluye importaciones y calentamiento, el recorrido los excluye.",
            "El perfil usa otro proceso. RSS usa VmHWM tras exec, "
            "una aproximación asíncrona de Linux.",
            "No se miden energía, coste monetario, VRAM ni transferencias CPU/GPU.",
        ],
    )
    atomic_json(args.output, report)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
