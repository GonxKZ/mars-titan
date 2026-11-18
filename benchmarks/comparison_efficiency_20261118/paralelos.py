"""Medir dos recorridos independientes dentro de una única reserva técnica de GPU."""

import gc
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import time
import traceback
from multiprocessing.connection import wait
from pathlib import Path

from configuracion import workdir

OUT = workdir()
GIB = 1024**3
MODELS = (("titans_mac", 42), ("mars_titan_m2", 44))


def worker(conn, model, seed):
    try:
        import recorrido as rr
        import torch

        rr.rw._cuda("cuda:0")
        assert torch.cuda.is_available()
        torch.cuda.set_device("cuda:0")
        rr.apply_kernel_policy("fp32_strict")
        guard = rr.perf._forbid_steps()
        with rr.tw.unfused_attention():
            src = rr.sources()
            spec = src["train"].specification()
            engine, _, _ = rr.build(model, src, spec, seed)
            initial = rr.parameter_digest(engine)
            conn.send(
                (
                    "ready",
                    {
                        "model": model,
                        "seed": seed,
                        "weights": initial,
                        "device": torch.cuda.get_device_name(0),
                    },
                )
            )
            while True:
                item = conn.recv()
                if item == "stop":
                    break
                mode, audit = item

                def gate():
                    conn.send(("warm", None))
                    if conn.recv() != "go":
                        raise RuntimeError("Barrera desordenada")

                point = rr.profile_slice(engine, spec, mode, audit, warm=2, until=8, gate=gate)
                if audit:
                    point["parity_sha256"] = hashlib.sha256(
                        json.dumps(point.pop("parity"), sort_keys=True, allow_nan=False).encode()
                    ).hexdigest()
                engine.audit = None
                src["train"].last_reader = None
                gc.collect()
                assert engine.optimizer.unchanged() and rr.parameter_digest(engine) == initial
                conn.send(("done", point))
        guard.remove()
    except BaseException:
        conn.send(("error", traceback.format_exc()))
    finally:
        conn.close()


class Capacity(Exception):
    pass


class Resources:
    def __init__(self):
        cg = Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip().lstrip("/")
        self.root = Path("/sys/fs/cgroup") / cg
        self.limit = int((self.root / "memory.max").read_text())
        self.reset()

    def reset(self):
        self.peak = 0
        self.anon = 0
        self.min_ram = 1 << 62
        self.min_gpu = 1 << 62
        self.last_gpu = 0.0

    def sample(self):
        value = int((self.root / "memory.current").read_text())
        self.peak = max(self.peak, value)
        values = dict(line.split() for line in (self.root / "memory.stat").read_text().splitlines())
        anon = int(values["anon"]) + int(values.get("shmem", 0))
        self.anon = max(self.anon, anon)
        avail = next(
            int(line.split()[1]) * 1024
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        )
        self.min_ram = min(avail, self.min_ram)
        if anon > self.limit - 448 * 1024**2 or avail < 4 * GIB:
            raise Capacity(
                f"Margen de RAM agotado: anónima {anon}, disponible {avail}, límite {self.limit}"
            )
        now = time.monotonic()
        if now - self.last_gpu > 1.0:
            q = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--id=0",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=4,
            )
            free = int(q.strip()) * 1024**2
            self.min_gpu = min(self.min_gpu, free)
            self.last_gpu = now
            if free < 768 * 1024**2:
                raise Capacity("Quedan menos de 768 MiB en la GPU")

    def data(self):
        return dict(
            cgroup_max_bytes=self.limit,
            cgroup_peak_bytes=self.peak,
            cgroup_peak_anon_bytes=self.anon,
            host_min_available_bytes=self.min_ram,
            device_min_free_sampled_bytes=self.min_gpu,
        )


def receive(pipes, stage, resources, limit):
    result = {}
    active = dict(pipes)
    until = time.monotonic() + limit
    while active:
        if time.monotonic() > until:
            raise TimeoutError(stage)
        resources.sample()
        ready = wait(list(active.values()), timeout=0.25)
        for pipe in ready:
            name = next(k for k, v in active.items() if v == pipe)
            kind, value = pipe.recv()
            if kind != "error" and kind == stage:
                result[name] = value
            else:
                raise RuntimeError(f"{name}/{stage}: {value}")
            del active[name]
    return result


def batch(pipes, mode, audit, parallel, res):
    res.reset()
    begin = time.perf_counter()
    results = {}
    groups = [pipes] if parallel else [{name: pipe} for name, pipe in pipes.items()]
    for group in groups:
        for pipe in group.values():
            pipe.send((mode, audit))
        receive(group, "warm", res, 95)
        for pipe in group.values():
            pipe.send("go")
        results.update(receive(group, "done", res, 95))
    complete_seconds = time.perf_counter() - begin
    timed = list(results.values())
    steady = (
        max(x["monotonic_finish"] for x in timed) - min(x["monotonic_start"] for x in timed)
        if parallel
        else sum(x["seconds"] for x in timed)
    )
    rows = sum(x["samples"] for x in timed)
    return dict(
        mode=mode,
        parallel=parallel,
        with_parity_audit=audit,
        pair_wall_seconds=complete_seconds,
        pair_steady_seconds=steady,
        total_scored_observations=rows,
        aggregate_rows_s=rows / steady,
        workers=results,
        resources=res.data(),
    )


def publish(result):
    tmp = OUT / "paralelos.partial"
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(OUT / "paralelos.json")


def main():
    if (OUT / "paralelos.json").exists() or (OUT / "paralelos.partial").exists():
        raise FileExistsError("El ensayo necesita una salida nueva")
    start = time.perf_counter()
    res = Resources()
    ctx = mp.get_context("spawn")
    pipes = {}
    procs = []
    result = dict(
        models=MODELS,
        number_real_optimizer_steps=0,
        mps=bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY")),
        ready={},
        pairs=[],
        complete=False,
    )
    try:
        for model, seed in MODELS:
            a, b = ctx.Pipe()
            proc = ctx.Process(target=worker, args=(b, model, seed))
            proc.start()
            b.close()
            procs.append(proc)
            pipes[model] = a
            result["ready"].update(receive({model: a}, "ready", res, 120))
        result["setup_seconds"] = time.perf_counter() - start
        publish(result)
        for mode in ("preparada", "original"):
            # La paridad de los dos procesos se contrasta aparte del cronometraje.
            baseline = None
            for audit, parallel in (
                (True, False),
                (True, True),
                (False, False),
                (False, True),
                (False, True),
                (False, False),
            ):
                point = batch(pipes, mode, audit, parallel, res)
                if audit:
                    hashes = {m: x["parity_sha256"] for m, x in point["workers"].items()}
                    if baseline is None:
                        baseline = hashes
                    elif hashes != baseline:
                        raise ValueError(
                            "La concurrencia cambió alguna emisión, gradiente o contador"
                        )
                result["pairs"].append(point)
                publish(result)
                print(
                    mode,
                    parallel,
                    audit,
                    point["pair_steady_seconds"],
                    point["pair_wall_seconds"],
                    flush=True,
                )
        result["complete"] = True
    except Capacity as e:
        result["capacity_stop"] = str(e)
        print(str(e), flush=True)
    except BaseException:
        result["error"] = traceback.format_exc()
        raise
    finally:
        for conn in pipes.values():
            try:
                conn.send("stop")
            except (OSError, EOFError):
                pass
        for proc in procs:
            proc.join(2)
            if proc.is_alive():
                proc.terminate()
                proc.join(4)
        for conn in pipes.values():
            conn.close()
        result["elapsed_seconds"] = time.perf_counter() - start
        publish(result)


if __name__ == "__main__":
    main()
