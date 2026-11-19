"""Medir ajustes con AdamW y guardar huellas de un checkpoint de ensayo independiente.

La copia científica se elige con PYTHONPATH. La autorización del ensayo vive en
WORK/admisiones/COMMIT. Las salidas se crean en WORK/ajustes/COMMIT, fuera de la cola.
"""

import argparse
import hashlib
import json
import os
import random
import struct
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.optim.optimizer import register_optimizer_step_post_hook

from mars_titan.training import comparison_campaign as campaign
from mars_titan.training import mars_titan_walk_forward as reader
from mars_titan.training import titans_walk_forward as core
from mars_titan.training.checkpoints import StopRequest


def state_fingerprint(tree):
    """Incluir tipos, formas, strides, ceros con signo y el contenido de cada campo."""
    digest = hashlib.sha256()
    tensors = 0
    elements = 0

    def frame(data):
        digest.update(struct.pack("!Q", len(data)))
        digest.update(data)

    def visit(obj):
        nonlocal tensors, elements
        frame(type(obj).__qualname__.encode())
        if isinstance(obj, torch.Tensor):
            frame(str((tuple(obj.shape), tuple(obj.stride()), str(obj.dtype))).encode())
            raw = obj.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy()
            frame(memoryview(raw))
            tensors += 1
            elements += obj.numel()
        elif isinstance(obj, dict):
            frame(struct.pack("!Q", len(obj)))
            for key in sorted(obj, key=lambda k: (type(k).__qualname__, str(k))):
                visit(key)
                visit(obj[key])
        elif isinstance(obj, (list, tuple)):
            frame(struct.pack("!Q", len(obj)))
            for item in obj:
                visit(item)
        elif isinstance(obj, float):
            frame(struct.pack("!d", obj))
        elif isinstance(obj, (type(None), bool, int, str, bytes)):
            frame(repr(obj).encode())
        else:
            raise ValueError(f"Tipo de checkpoint sin huella definida: {type(obj).__qualname__}")

    visit(tree)
    return dict(sha256=digest.hexdigest(), tensors=tensors, tensor_values=elements)


def measured_fit(job, model, recipe, rule, source, directory, requested_updates, flags):
    view = source / "cohorts" / str(job.year) / "manifest.json"
    updates = 0
    first = last = None
    with StopRequest() as stop, core.unfused_attention(), flags:

        def record_step(optimizer, positional, keywords):
            nonlocal updates, first, last
            updates += 1
            if updates == 1:
                torch.cuda.synchronize()
                first = time.perf_counter()
            if updates == requested_updates:
                torch.cuda.synchronize()
                last = time.perf_counter()
                stop.request_stop()

        hook = register_optimizer_step_post_hook(record_step)
        start = time.perf_counter()
        try:
            common = dict(
                seed=job.seed, output=directory, device="cuda:0", stop=stop, stopping=rule
            )
            if job.runner == "fit":
                report = core.run_titans_window(
                    view,
                    core.view_protocol(view),
                    job.fold,
                    recipe,
                    variant=model["variant"],
                    predictor_changes=model.get("predictor_changes"),
                    **common,
                )
            else:
                report = reader.run_mars_titan_window(
                    view,
                    source / "runs" / job.needs,
                    recipe,
                    components=dict(model["components"]),
                    search_case=model["search_case"],
                    **common,
                )
        finally:
            hook.remove()
    elapsed = time.perf_counter() - start
    if report["status"] != "paused" or updates != requested_updates or last is None:
        raise ValueError("El ajuste no llegó al paso solicitado")
    return elapsed, last - first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-state", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--machine", choices=campaign.MACHINES, default="mars_titan")
    parser.add_argument("--model", choices=("titans_mac", "mars_titan_m3_k4"), required=True)
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or not args.work.is_absolute() or not 2000 <= args.year <= 2023:
        parser.error("Se necesita una salida nueva, un trabajo absoluto y un origen histórico")
    work = args.work.resolve(strict=True)
    source = args.source_state.resolve(strict=True)
    output = args.output.resolve()
    if work.is_relative_to(source) or source.is_relative_to(work) or output.is_relative_to(source):
        parser.error("La campaña y el ensayo deben vivir en directorios independientes")
    root = Path(core.__file__).resolve().parents[3]
    commit = campaign.code_commit(root)
    if not commit or campaign.code_changes(root) is not None:
        parser.error("El runtime científico debe ser una revisión fija")
    declaration, digest = campaign.load_declaration(root=root)
    layout = campaign.Layout(work / "admisiones" / commit)
    if not campaign.authorized(layout, digest, commit=commit, changes=None):
        parser.error("Falta la autorización de esta revisión en el directorio de ensayo")
    shared_indices = os.environ.get("MARS_TITAN_SHARED_INDICES")
    os.environ.update(campaign._environment(layout, declaration))
    if shared_indices:
        os.environ["MARS_TITAN_SHARED_INDICES"] = shared_indices
    flags = nullcontext()
    cache = None
    if args.packed:
        from mars_titan.memory import training_event_cache as cache
        from mars_titan.models.titans import host_finite

        if not os.environ.get(cache.ENV):
            parser.error("Hay que indicar el directorio de entradas preparado")
        cache.reset_usage()
        if args.model == "titans_mac":
            flags = host_finite.on_host()
    else:
        os.environ["MARS_TITAN_TRAIN_ARCHIVE"] = ""
        os.environ["MARS_TITAN_CUDA_HOST_FINITE"] = "0"
    key = f"{args.model}/{args.year}/seed-{args.seed}"
    job = next((j for j in campaign.plan(declaration, args.machine) if j.key == key), None)
    if job is None:
        parser.error("El trabajo seleccionado no pertenece al equipo")
    model = declaration["models"][job.model]
    directory = work / "ajustes" / commit / job.key
    fit_path = directory / "fit/run.json"
    begin = json.loads(fit_path.read_text())["global_step"] if fit_path.exists() else 0
    if args.step <= begin:
        parser.error("El paso de parada debe superar al del checkpoint inicial")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    elapsed, measured = measured_fit(
        job,
        model,
        root / declaration["recipes"][model["recipe"]],
        campaign.job_rule(declaration, model),
        source,
        directory,
        args.step - begin,
        flags,
    )
    checkpoints = directory / "fit/checkpoints"
    ref = json.loads((checkpoints / "latest.json").read_text())["latest"][0]
    path = checkpoints / ref["name"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != ref["sha256"]:
            raise ValueError("El contenido no corresponde a la huella del checkpoint")
    state = torch.load(path, map_location="cpu", weights_only=True)["state"]
    if state["global_step"] != args.step or ref["global_step"] != args.step:
        raise ValueError("El checkpoint pertenece a otro paso")
    usage = cache.receipt() if cache else {}
    if args.packed and (not usage.get("packed_passes") or usage.get("builds", 0)):
        raise ValueError("El ajuste necesita la copia ya preparada")
    result = dict(
        model=args.model,
        year=args.year,
        seed=args.seed,
        commit=commit,
        from_step=begin,
        to_step=args.step,
        updates=args.step - begin,
        cursor=state["cursor"],
        checkpoint_sha256=ref["sha256"],
        first_to_last_update_seconds=measured,
        entry_to_checkpoint_seconds=elapsed,
        usage=usage,
        fields={k: state_fingerprint(v) for k, v in state.items()},
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as dest:
        json.dump(result, dest, indent=2)
        dest.write("\n")
    print(args.model, args.step, round(elapsed, 2), flush=True)


if __name__ == "__main__":
    main()
