"""Contrastar revisiones científicas en procesos nuevos sobre un mismo checkpoint."""

import argparse
import json
import os
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import recorrido
import torch
from entradas_campana import sources

from mars_titan.data.storage import atomic_json


def measure(model, year, seed, *, packed, audit):
    src = sources(year, model, seed)
    train = src["train"]
    spec = train.specification()
    flags = nullcontext()
    cache = None
    if packed:
        from mars_titan.memory import training_event_cache as cache
        from mars_titan.models.titans import host_finite

        src["train"] = cache.PackedTrainingSource(train.dataset, train.path, os.environ[cache.ENV])
        if model == "titans_mac":
            flags = host_finite.on_host()
        cache.reset_usage()
    segments = 16 if audit else 68
    start = int((0.9 if audit else 0.8) * len(train.metadata["groups"]))
    count = 144 if audit else 576
    with flags, recorrido.tw.unfused_attention():
        begin = time.perf_counter()
        engine, _, _ = recorrido.build(model, src, spec, seed, year=year)
        weight = recorrido.parameter_digest(engine)
        case = recorrido.profile_slice(
            engine,
            spec,
            "original",
            audit,
            source_range=(start, count),
            rng_seed=seed,
            until=segments,
        )
        build_and_pass = time.perf_counter() - begin
        if weight != recorrido.parameter_digest(engine):
            raise ValueError("El ensayo cambió los pesos seleccionados")
    usage = cache.receipt() if packed else {}
    if packed and (not usage.get("packed_passes") or usage.get("build_seconds")):
        raise ValueError(
            "Hay que preparar previamente la copia para separar su coste de construcción"
        )
    runtime = Path(recorrido.tw.__file__).resolve().parents[3]
    commit = subprocess.check_output(["git", "-C", runtime, "rev-parse", "HEAD"], text=True).strip()
    return dict(
        commit=commit,
        model=model,
        year=year,
        seed=seed,
        gpu=torch.cuda.get_device_name(0),
        torch=str(torch.__version__),
        packed=packed,
        host_finite=packed and model == "titans_mac",
        real_optimizer_steps=0,
        weights_sha256=weight,
        input_index_sha256=train.identity,
        usage=usage,
        build_and_pass_seconds=build_and_pass,
        case=case,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("titans_mac", "mars_titan_m2"), required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Cada ejecución necesita una salida nueva")
    if not 2000 <= args.year <= 2023:
        parser.error("El ensayo necesita una cohorte histórica de desarrollo o confirmación")
    recorrido.rw._cuda("cuda:0")
    torch.cuda.set_device(0)
    recorrido.apply_kernel_policy("fp32_strict")
    guard = recorrido.perf._forbid_steps()
    try:
        result = measure(args.model, args.year, args.seed, packed=args.packed, audit=args.audit)
    finally:
        guard.remove()
    atomic_json(args.output, result)
    print(json.dumps({"modelo": args.model, "segundos_recorrido": result["case"]["pass_seconds"]}))


if __name__ == "__main__":
    main()
