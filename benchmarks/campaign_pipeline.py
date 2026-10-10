"""Medir la lectura y las ranuras GPU de la campaña con máscaras sin ajustar ningún modelo.

Subórdenes:

- `reader`: filas por segundo del lector del corpus sin modelo.
- `steps`: recorrido real de las referencias neuronales en `cuda:0` (lectura, copia, forward,
  pérdida y backward) sin pasos de optimizador. Separa la espera al lector del tramo GPU.
- `slots`: uno, dos y tres trabajos a la vez, cada uno en un proceso nuevo con el entorno de
  su ranura. Cada trabajo resume sus pérdidas y gradientes en una huella para comprobar que
  da los mismos bits acompañado que solo.
- `chronological`: filas por segundo del lector por instantes de las familias de memoria.
- `cpu-load`: procesos de Python ocupados, para medir cuánto pierde la campaña sin CPU libre.

La tubería se elige con las mismas variables que la campaña (`MARS_TITAN_DECODE_WORKERS`,
`MARS_TITAN_PREFETCH_BATCHES` y `MARS_TITAN_GROUP_CACHE_MIB`). Así el mismo guion sirve para
una revisión anterior fijada con PYTHONPATH, que simplemente las ignora. Con
`MARS_TITAN_DIGEST_CACHE` la rama reutiliza las huellas SHA-256 de la vista, y la apertura se
mide aparte porque una revisión sin esa caché vuelve a leer todos los archivos. Las medidas
GPU usan FP32 estricto y algoritmos deterministas, y ningún subcomando da pasos de optimizador.
"""

import argparse
import hashlib
import itertools
import json
import multiprocessing
import os
import resource
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

CAMPAIGN = "configs/baselines/historical-masked-campaign-a.json"
TITANS_RECIPE = "configs/titans/chronological-training-historical-masked.json"
MIB = 1024**2
_SPAWN = multiprocessing.get_context("spawn")


def _context():
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    ).stdout.strip()
    pipeline = {key: value for key, value in os.environ.items() if key.startswith("MARS_TITAN_")}
    return dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=commit or None,
        environment=pipeline,
        load=Path("/proc/loadavg").read_text().split()[:3],
        optimizer_steps=0,
    )


def _peak_rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def _campaign():
    from mars_titan.training.campaign_plan import load_campaign

    campaign = load_campaign(CAMPAIGN)
    cases = {name: case for arm in campaign["neural"]["candidates"].values() for name, case in arm}
    return campaign, cases


def _opened(view, campaign):
    from mars_titan.training.reference_run import configured_corpus

    started = time.perf_counter()
    dataset = configured_corpus(view, input_policy=campaign["input_policy"])
    return dataset, time.perf_counter() - started


def _train_batches(dataset, campaign, seed):
    size = campaign["neural"]["batch_size"]
    return iter(dataset.batches(partition="train", batch_size=size, epoch=0, seed=seed))


def reader(args):
    campaign, _ = _campaign()
    dataset, opened = _opened(args.view, campaign)
    source = _train_batches(dataset, campaign, 42)
    next(source)  # El primer lote incluye el arranque de la tubería y se deja fuera.
    rows, cpu, start = 0, time.process_time(), time.perf_counter()
    for _ in range(args.batches):
        rows += len(next(source)["target"])
    seconds, cpu = time.perf_counter() - start, time.process_time() - cpu
    getattr(source, "close", lambda: None)()
    return dict(
        open_s=opened,
        rows=rows,
        seconds=seconds,
        rows_s=rows / seconds,
        cpu_us_row=cpu / rows * 1e6,
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _strict_cuda():
    """Preparar `cuda:0` con FP32 estricto y deterministas. Sin GPU falla en vez de usar CPU."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from mars_titan.data.embeddings import require_cuda
    from mars_titan.training.campaign_slots import strict_fp32

    strict_fp32()
    torch.use_deterministic_algorithms(True)
    return torch, require_cuda()


def _loss(model, batch, device):
    import torch

    from mars_titan.models.quantile_head import pinball_loss
    from mars_titan.training.reference_run import _forward

    target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
    loss = pinball_loss(_forward(model, batch, device), target)
    loss.backward()
    if not torch.isfinite(loss).item():  # La misma comprobación por lote que el entrenador.
        raise ValueError("Pérdida no finita")
    return loss.detach()


def steps(args):
    torch, device = _strict_cuda()
    from mars_titan.training.campaign_throughput import _forbid_steps, _reference

    campaign, cases = _campaign()
    dataset, opened = _opened(args.view, campaign)
    guard, results = _forbid_steps(), {}
    for name in args.cases:
        model, _ = _reference(cases[name], dataset)
        model = model.to(device).train(True)
        torch.cuda.reset_peak_memory_stats(0)
        source = _train_batches(dataset, campaign, cases[name]["seed"])
        events, waited, rows = [], 0.0, 0
        for index in range(args.warmup + args.batches):
            if index == args.warmup:
                torch.cuda.synchronize(0)
                start = time.perf_counter()
            before = time.perf_counter()
            batch = next(source)
            if index >= args.warmup:
                waited += time.perf_counter() - before
            begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            begin.record()
            _loss(model, batch, device)
            model.zero_grad(set_to_none=True)
            end.record()
            if index >= args.warmup:
                events.append((begin, end))
                rows += len(batch["target"])
        torch.cuda.synchronize(0)
        wall = time.perf_counter() - start
        getattr(source, "close", lambda: None)()
        busy = sum(begin.elapsed_time(end) for begin, end in events) / 1000
        results[name] = dict(
            rows=rows,
            rows_s=rows / wall,
            reader_wait_fraction=waited / wall,
            gpu_busy_fraction=busy / wall,
            peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
        )
        print(name, json.dumps(results[name]), flush=True)
    guard.remove()
    return dict(
        open_s=opened,
        results=results,
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        tf32=[torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32],
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _probe(view, name, batches, environment, vram_bytes, queue, barrier):
    """Trabajo de una ranura. Las pérdidas y uno de cada ocho gradientes forman su huella."""
    os.environ.update(environment)
    torch, device = _strict_cuda()
    from mars_titan.training.campaign_resources import CONTEXT_MIB
    from mars_titan.training.campaign_slots import bound_vram
    from mars_titan.training.campaign_throughput import _forbid_steps, _reference

    bound_vram(vram_bytes - CONTEXT_MIB * MIB)
    campaign, cases = _campaign()
    dataset, _ = _opened(view, campaign)
    model, _ = _reference(cases[name], dataset)
    model = model.to(device).train(True)
    guard, digest, rows, losses = _forbid_steps(), hashlib.sha256(), 0, []
    source = _train_batches(dataset, campaign, cases[name]["seed"])
    first = next(source)
    torch.cuda.synchronize(0)
    barrier.wait()  # Todos empiezan a la vez y con la vista ya abierta.
    start = time.perf_counter()
    for index, batch in enumerate(itertools.chain([first], source)):
        if index == batches:
            break
        losses.append(_loss(model, batch, device))
        if index % 8 == 0:
            gradients = [p.grad.reshape(-1) for p in model.parameters() if p.grad is not None]
            digest.update(torch.cat(gradients).cpu().numpy().tobytes())
        model.zero_grad(set_to_none=True)
        rows += len(batch["target"])
    torch.cuda.synchronize(0)
    end = time.perf_counter()
    digest.update(torch.stack(losses).cpu().numpy().tobytes())
    guard.remove()
    queue.put(
        dict(
            name=name,
            rows=rows,
            start=start,
            end=end,
            digest=digest.hexdigest(),
            peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(0),
            peak_rss_bytes=_peak_rss_bytes(),
        )
    )


def _together(args, names, vram_bytes):
    from mars_titan.training.campaign_resources import Execution, JobResources
    from mars_titan.training.campaign_slots import slot_environment
    from mars_titan.training.input_pipeline import PipelineOptions

    execution = Execution(
        gpu_slots=max(2, len(names)),
        mps=args.mps,
        mps_pipe_directory=args.mps_pipe_directory if args.mps else None,
        threads_per_job=args.threads,
        pipeline=PipelineOptions.from_environment(),
    )
    environment = slot_environment(execution, JobResources("cuda", vram_bytes, 0))
    environment["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    queue, barrier = _SPAWN.Queue(), _SPAWN.Barrier(len(names))
    processes = [
        _SPAWN.Process(
            target=_probe,
            args=(args.view, name, args.batches, environment, vram_bytes, queue, barrier),
        )
        for name in names
    ]
    for process in processes:
        process.start()
    jobs = [queue.get() for _ in processes]
    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError(f"Una sonda terminó con el código {process.exitcode}")
    first, last = min(job["start"] for job in jobs), max(job["end"] for job in jobs)
    return dict(
        jobs=names,
        aggregate_rows_s=sum(job["rows"] for job in jobs) / (last - first),
        per_job=jobs,
    )


def slots(args):
    """Cada caso solo y después los dos y los tres primeros a la vez, con la misma VRAM."""
    vram_bytes = args.vram_mib * MIB
    runs, solo = [], {}
    for name in args.cases:
        run = _together(args, [name], vram_bytes)
        solo[name] = run["per_job"][0]["digest"]
        runs.append(run)
        print("solo", name, round(run["aggregate_rows_s"]), flush=True)
    for count in range(2, len(args.cases) + 1):
        run = _together(args, args.cases[:count], vram_bytes)
        run["bit_equal_to_solo"] = all(job["digest"] == solo[job["name"]] for job in run["per_job"])
        runs.append(run)
        print(count, round(run["aggregate_rows_s"]), run["bit_equal_to_solo"], flush=True)
    return dict(mps=args.mps, vram_bytes_per_job=vram_bytes, runs=runs)


def chronological(args):
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.memory import financial_observations
    from mars_titan.training import titans_walk_forward as titans
    from mars_titan.training.campaign_throughput import _view_fold
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.financial_run import load_recipe

    if args.lru:
        # Con un horizonte imposible ningún activo cuenta como ausente y la caché descarta
        # siempre el grupo usado hace más tiempo, como una LRU.
        financial_observations._BlockReader.STALE_EVENTS = -(10**12)
    _, document = load_recipe(TITANS_RECIPE)
    dataset = CorpusDataset(args.view, input_policy=HISTORICAL_MASKED)
    phases = titans.window_phases(
        _view_fold(dataset), titans.walk_forward_options(document)["warmup_months"]
    )
    source = titans._sources(dataset, {"train": phases["train"]}, args.index)["train"]
    total = len(source.metadata["groups"])
    start = total + args.start if args.start < 0 else args.start
    stream = source.batched_events(
        start_cursor=start, block_rows=args.block_rows, max_cached_bytes=args.cache_mib * MIB
    )
    began = time.perf_counter()
    for _ in zip(range(args.warm), stream, strict=False):
        pass
    warm_s, rows, events = time.perf_counter() - began, 0, 0
    cpu, began = time.process_time(), time.perf_counter()
    for event in stream:
        rows += sum(len(batch["sample_ids"]) for batch in event.inputs)
        events += 1
        if events >= args.events:
            break
    seconds, cpu = time.perf_counter() - began, time.process_time() - cpu
    stream.close()
    state = getattr(source, "last_reader", None)
    return dict(
        total_events=total,
        start=start,
        warm_events=args.warm,
        warm_s=warm_s,
        cache_mib=args.cache_mib,
        policy="lru" if args.lru else "default",
        events=events,
        rows=rows,
        rows_s=rows / seconds,
        cpu_us_row=cpu / rows * 1e6,
        redecoded_groups=getattr(state, "redecoded_groups", None),
        peak_cached_bytes=getattr(state, "peak_cached_bytes", None),
        peak_rss_bytes=_peak_rss_bytes(),
    )


def _burn(seconds, parent):
    end = time.monotonic() + seconds
    while time.monotonic() < end and os.getppid() == parent:
        sum(value * value for value in range(20_000))


def cpu_load(args):
    """Ocupar `processes` núcleos con Python puro. Cada proceso termina si muere su padre."""
    workers = [
        _SPAWN.Process(target=_burn, args=(args.seconds, os.getpid()), daemon=True)
        for _ in range(args.processes)
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    measured = {}
    for name in ("reader", "steps", "slots", "chronological"):
        measured[name] = commands.add_parser(name)
        measured[name].add_argument("view", type=Path)
        measured[name].add_argument("--output", type=Path)
    measured["reader"].add_argument("--batches", type=int, default=400)
    measured["steps"].add_argument("--cases", nargs="+", required=True)
    measured["steps"].add_argument("--batches", type=int, default=150)
    measured["steps"].add_argument("--warmup", type=int, default=5)
    measured["slots"].add_argument("--cases", nargs="+", required=True)
    measured["slots"].add_argument("--batches", type=int, default=600)
    measured["slots"].add_argument("--vram-mib", type=int, default=1536)
    measured["slots"].add_argument("--threads", type=int, default=2)
    measured["slots"].add_argument("--mps", action="store_true")
    measured["slots"].add_argument("--mps-pipe-directory", default="/run/user/1000/nvidia-mps")
    chrono = measured["chronological"]
    chrono.add_argument("--index", type=Path, required=True)
    chrono.add_argument("--events", type=int, default=100)
    chrono.add_argument("--start", type=int, default=0, help="negativo para contar desde el final")
    chrono.add_argument("--warm", type=int, default=10)
    chrono.add_argument("--block-rows", type=int, default=128)
    chrono.add_argument("--cache-mib", type=int, default=4096)
    chrono.add_argument("--lru", action="store_true")
    load = commands.add_parser("cpu-load")
    load.add_argument("processes", type=int)
    load.add_argument("seconds", type=float)
    return parser


def main():
    args = _parser().parse_args()
    if args.command == "cpu-load":
        cpu_load(args)
        return
    measure = dict(reader=reader, steps=steps, slots=slots, chronological=chronological)
    report = dict(_context(), command=args.command, view=str(args.view))
    report.update(measure[args.command](args))
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output is None:
        print(text)
    else:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
