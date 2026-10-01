"""Medir la acumulación dentro de la evaluación CPU con sesiones repetidas y únicas."""

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
from evaluation_io import DiagnosticRows

from mars_titan.data.storage import atomic_json, sha256


def worker(args):
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    engine = importlib.import_module("mars_titan.training.reference_run")
    dataset = DiagnosticRows(16384)
    if args.layout == "unique":
        dataset.times = 1_546_300_800_000_000 + np.arange(dataset.count, dtype=np.int64)
    linear = torch.nn.Linear(16, 1)
    with torch.no_grad():
        linear.weight.copy_(torch.arange(16, dtype=torch.float32).reshape(1, 16) / 64)
        linear.bias.fill_(0.125)

    class Predictor(torch.nn.Module):
        def forward(self, inputs):
            return linear(inputs["values"]).squeeze(-1)

    model = Predictor()
    output = args.work / f"{args.layout}-{args.batch_size}.parquet" if args.persist else None

    def evaluate():
        with (
            patch.object(
                engine,
                "_inputs",
                lambda batch, _device: {"values": torch.from_numpy(batch["inputs"]["values"])},
            ),
            patch.object(torch.cuda, "synchronize", lambda _device: None),
        ):
            return engine._evaluate(model, dataset, args.batch_size, destination=output)

    expected = evaluate()
    durations = []
    for _ in range(args.repetitions):
        started = time.perf_counter()
        result = evaluate()
        durations.append(time.perf_counter() - started)
        for key in expected:
            if (
                key not in {"elapsed_seconds", "samples_per_second"}
                and result[key] != expected[key]
            ):
                raise ValueError(f"La repetición altera {key}")
    status = dict(line.split(":", 1) for line in Path("/proc/self/status").read_text().splitlines())
    peak_rss = int(status["VmHWM"].split()[0]) * 1024
    profiler = cProfile.Profile()
    profiler.runcall(evaluate)
    profile = [
        dict(function=name, calls=values[1], self_seconds=values[2], total_seconds=values[3])
        for (file, _line, name), values in pstats.Stats(profiler).stats.items()
        if Path(file).name == "session_metrics.py"
    ]
    print(
        json.dumps(
            dict(
                layout=args.layout,
                batch_size=args.batch_size,
                persist=args.persist,
                rows=dataset.count,
                repetitions=args.repetitions,
                metrics={
                    k: v
                    for k, v in result.items()
                    if k not in {"elapsed_seconds", "samples_per_second"}
                },
                durations_seconds=durations,
                median_seconds=float(np.median(durations)),
                std_seconds=float(np.std(durations, ddof=1)),
                samples_per_second=dataset.count / float(np.median(durations)),
                worker_peak_rss_bytes=peak_rss,
                input_bytes=dataset.values.nbytes + dataset.target.nbytes + dataset.times.nbytes,
                prediction_sha256=sha256(output) if output else None,
                prediction_bytes=output.stat().st_size if output else 0,
                profile=profile,
                torch_version=torch.__version__,
            )
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--layout", choices=("cohorts", "unique"))
    parser.add_argument("--batch-size", type=int, choices=(31, 32, 33, 64, 65, 128, 129, 512, 4096))
    parser.add_argument("--persist", action="store_true")
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.output is None or not 3 <= args.repetitions <= 10:
        raise ValueError("Indica una salida y entre tres y diez repeticiones")
    args.work.mkdir(parents=True, exist_ok=True)
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES="",
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    runs = []
    for layout in ("cohorts", "unique"):
        for batch_size in (32, 512, 4096):
            for persist in (False, True):
                command = [
                    sys.executable,
                    __file__,
                    "--worker",
                    "--layout",
                    layout,
                    "--batch-size",
                    str(batch_size),
                    "--work",
                    str(args.work),
                    "--repetitions",
                    str(args.repetitions),
                ]
                if persist:
                    command.append("--persist")
                started = time.perf_counter()
                run = json.loads(subprocess.check_output(command, env=env, text=True, timeout=90))
                run["total_process_seconds"] = time.perf_counter() - started
                runs.append(run)
                print(layout, batch_size, persist, run["median_seconds"], flush=True)
    cpu = subprocess.check_output(["rg", "-m", "1", "^model name", "/proc/cpuinfo"], text=True)
    atomic_json(
        args.output,
        dict(
            measured_at=datetime.now(UTC).isoformat(),
            domain="technical",
            device="cpu",
            final_test_opened=False,
            hardware=dict(cpu=cpu.strip(), os=platform.platform()),
            software=dict(python=platform.python_version(), numpy=np.__version__),
            numerical_threads=1,
            warmup_runs=1,
            code_sha256={
                name: sha256(Path(name))
                for name in (
                    "src/mars_titan/evaluation/session_metrics.py",
                    "src/mars_titan/training/reference_run.py",
                    "benchmarks/session_metrics.py",
                    "benchmarks/evaluation_io.py",
                )
            },
            runs=runs,
            limits=[
                "Modelo lineal mínimo y datos sintéticos en CPU, sin entrenamiento ni CUDA.",
                "Repeticiones en un proceso tras un calentamiento, sin vaciar cachés.",
                "El tiempo de proceso incluye importaciones y todas las ejecuciones.",
                "VmHWM se lee antes del perfil. Es una aproximación asíncrona de Linux tras exec.",
                "No se miden energía, coste monetario, VRAM ni transferencias CPU/GPU.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
