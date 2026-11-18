"""Distinguir tiempo de kernels, copias y lanzamientos dentro de la ventana marcada."""

import argparse
import json
from collections import defaultdict

from configuracion import workdir

BASE = workdir()


def intervals(values):
    ordered = sorted(values)
    if not ordered:
        return 0.0
    start, end = ordered[0]
    total = 0.0
    for a, b in ordered[1:]:
        if a > end:
            total += end - start
            start, end = a, b
        else:
            end = max(end, b)
    return total + end - start


def summary(model):
    events = json.loads((BASE / f"{model}-kernels.json").read_text())["traceEvents"]
    matches = [
        e
        for e in events
        if e.get("name") == "auditoria.tramo_cuda"
        and e.get("ph") == "X"
        and e.get("cat") not in ("gpu_user_annotation", "kernel")
    ]
    if len(matches) != 1:
        raise ValueError("Falta una ventana de perfil inequívoca")
    begin = matches[0]["ts"]
    duration = matches[0]["dur"]
    finish = begin + duration
    times = defaultdict(list)
    names = defaultdict(lambda: defaultdict(list))
    for e in events:
        ty = e.get("cat", "")
        if e.get("ph") != "X" or ty not in (
            "kernel",
            "gpu_memcpy",
            "gpu_memset",
            "cuda_runtime",
            "cuda_driver",
        ):
            continue
        a = max(e["ts"], begin)
        b = min(e["ts"] + e["dur"], finish)
        if b <= a:
            continue
        times[ty].append((a, b))
        names[ty][e["name"]].append(b - a)
    if not times["kernel"]:
        raise ValueError("El trazador no devolvió actividad de kernels CUDA")
    out = dict(capture_model=model, wall_ms=duration / 1000, categories={})
    for k, entries in times.items():
        unit = intervals(entries)
        top = sorted(((sum(v), n, len(v)) for n, v in names[k].items()), reverse=True)
        out["categories"][k] = dict(
            calls=len(entries),
            sum_ms=sum(b - a for a, b in entries) / 1000,
            union_ms=unit / 1000,
            union_percent=100 * unit / duration,
            top=[dict(name=n, milliseconds=t / 1000, calls=c) for t, n, c in top[:10]],
        )
    (BASE / f"{model}-perfil-resumen.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        model,
        duration / 1000,
        out["categories"]["kernel"]["calls"],
        out["categories"]["kernel"]["union_percent"],
    )
    return out


def nsys_run(model):
    import recorrido as rr
    import torch

    rr.rw._cuda("cuda:0")
    assert torch.cuda.is_available()
    torch.cuda.set_device("cuda:0")
    rr.apply_kernel_policy("fp32_strict")
    guard = rr.perf._forbid_steps()

    class Trigger:
        def start(self):
            torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push("auditoria.tramo_cuda")

        def finish(self):
            torch.cuda.nvtx.range_pop()
            torch.cuda.cudart().cudaProfilerStop()

    try:
        with rr.tw.unfused_attention():
            src = rr.sources()
            spec = src["train"].specification()
            engine, _, _ = rr.build(model, src, spec)
            before = rr.parameter_digest(engine)
            result = rr.profile_slice(
                engine, spec, "preparada", False, warm=2, until=4, trace=Trigger()
            )
            assert rr.parameter_digest(engine) == before
            print(json.dumps(result))
    finally:
        guard.remove()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--nsys-run", action="store_true")
    args = p.parse_args()
    nsys_run(args.model) if args.nsys_run else summary(args.model)
