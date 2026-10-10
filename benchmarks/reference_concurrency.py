"""Caudal agregado de k procesos que ajustan referencias a la vez en la GPU, sin MPS.

Cada proceso hijo carga los lotes guardados por `reference_kernels.py materialize`,
construye su caso en FP32 estricto, calienta y espera a una hora de inicio común. Después
repite forward, pinball y backward sin optimizador durante un tiempo fijo y devuelve sus
filas por segundo, sus picos de memoria y la memoria del proceso según nvidia-smi.

    PYTHONPATH=src python benchmarks/reference_concurrency.py lotes.pt --cases gru-10 --jobs 1 2 3
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def child(args):
    import torch
    from reference_kernels import (
        CASES,
        apply_precision,
        build,
        process_mib,
        resident,
        stored_batches,
        train_step,
    )

    apply_precision("strict")
    device = torch.device("cuda:0")
    saved, loaded = stored_batches(args.batches, args.batch_size)
    batches = [resident(batch, device) for batch in loaded]
    size = args.batch_size
    dimensions = {k: v.shape[-1] for k, v in loaded[0]["inputs"].items()}
    model = build(dict(CASES)[args.child], dimensions, saved["context"], size).to(device)
    model.train()
    for batch in batches[:3]:
        train_step(model, batch)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    while time.time() < args.start_at:
        time.sleep(0.01)
    rows, steps, start = 0, 0, time.perf_counter()
    while time.perf_counter() - start < args.seconds:
        train_step(model, batches[steps % len(batches)])
        rows, steps = rows + size, steps + 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(
        json.dumps(
            dict(
                samples_per_second=rows / elapsed,
                steps=steps,
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                process_mib=process_mib(),
            )
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batches", type=Path)
    parser.add_argument("--cases", nargs="+", default=[])
    parser.add_argument("--jobs", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seconds", type=float, default=8.0)
    parser.add_argument("--setup-seconds", type=float, default=40.0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child")
    parser.add_argument("--start-at", type=float)
    args = parser.parse_args()
    if args.child:
        return child(args)
    results = {}
    for name in args.cases:
        for jobs in args.jobs:
            start_at = time.time() + args.setup_seconds
            command = [sys.executable, __file__, str(args.batches), "--child", name]
            command += ["--start-at", str(start_at), "--seconds", str(args.seconds)]
            command += ["--batch-size", str(args.batch_size)]
            processes = [
                subprocess.Popen(command, stdout=subprocess.PIPE, text=True) for _ in range(jobs)
            ]
            records = []
            for process in processes:
                output, _ = process.communicate()
                if process.returncode:
                    raise RuntimeError(f"{name} con {jobs} procesos terminó con error")
                records.append(json.loads(output.strip().splitlines()[-1]))
            results[f"{name}:{jobs}"] = dict(
                case=name,
                jobs=jobs,
                aggregate_samples_per_second=sum(r["samples_per_second"] for r in records),
                processes=records,
            )
            print(json.dumps(results[f"{name}:{jobs}"]), flush=True)
    if args.output:
        args.output.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
