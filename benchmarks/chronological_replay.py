"""Caudal, VRAM y gradientes de un brazo cronológico sin pasos de optimizador.

Reutiliza los constructores de `campaign_throughput`, que son los mismos antes y después de
un cambio: intercepta `_chronological` para obtener el constructor del brazo pedido y
recorre `_train_pass` desde un evento intermedio del tramo de ajuste. Las etiquetas cuyo
pronóstico no se emitió en este recorrido se descartan, como en un inicio a mitad de fase.
El optimizador es el sustituto de `campaign_throughput`, que solo cuenta llamadas, y al
final se comprueba que ningún parámetro ha cambiado.

Con `--events` se repiten los eventos guardados por `chronological_events.py` en lugar de
leer el corpus, de modo que la medida aísla el cálculo. Sin esa opción mide también el
lector. `chronological_replay_patches.py` acota el tramo y reutiliza las huellas de los
artefactos entre procesos, solo para medir.

Uso: chronological_replay.py VIEW WORK --family F --arm A [--option k=json]
     [--predictor k=json] [--recipe k=json] [--readout k=json] [--start-fraction f]
     [--warmup W] [--segments N] [--events eventos.pkl] [--capture out.pt]
     [--output out.jsonl] [--barrier dir --jobs k] [--label txt] [--strict]
"""

import argparse
import copy
import json
import os
import resource
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument("view", type=Path)
parser.add_argument("work", type=Path)
parser.add_argument(
    "--family", choices=["titans", "cm_core", "readout", "cm_readout", "candidate"], required=True
)
parser.add_argument("--arm", required=True)
parser.add_argument("--option", action="append", default=[])
parser.add_argument("--predictor", action="append", default=[])
parser.add_argument("--recipe", action="append", default=[])
parser.add_argument("--readout", action="append", default=[])
parser.add_argument("--start-fraction", type=float, default=0.5)
parser.add_argument("--warmup", type=int, default=3)
parser.add_argument("--segments", type=int, default=10)
parser.add_argument("--capture", type=Path)
parser.add_argument("--output", type=Path)
parser.add_argument("--barrier", type=Path)
parser.add_argument("--jobs", type=int, default=1)
parser.add_argument("--label", default="")
parser.add_argument("--strict", action="store_true")
parser.add_argument("--events", type=Path)
args = parser.parse_args()
if args.strict:
    # FP32 estricto también en la base, que todavía no declara la política en la receta.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")


def pairs(items):
    return {key: json.loads(value) for key, value in (item.split("=", 1) for item in items)}


options, predictor_over = pairs(args.option), pairs(args.predictor)
recipe_over, readout_over = pairs(args.recipe), pairs(args.readout)

sys.path.insert(0, str(Path(__file__).parent))
import chronological_replay_patches  # noqa: F401,E402

from mars_titan.training import campaign_throughput as ct  # noqa: E402
from mars_titan.training import financial_run as fr  # noqa: E402
from mars_titan.training import mars_titan_run as mr  # noqa: E402
from mars_titan.training import titans_walk_forward as titans  # noqa: E402
from mars_titan.training.campaign_extensions import (  # noqa: E402
    extended_campaign,
    load_extensions,
)
from mars_titan.training.campaign_plan import load_campaign  # noqa: E402

# Recetas con las sobrescrituras pedidas (lote del predictor, bloques, precisión, etc.).
_load_titans, _load_readout = fr.load_recipe, mr.load_recipe


def _titans_recipe(path):
    first, document = _load_titans(path)
    document = copy.deepcopy(document)
    document["predictor"].update(predictor_over)
    document["recipe"].update(recipe_over)
    return first, document


def _readout_recipe(path):
    document = copy.deepcopy(_load_readout(path))
    document["recipe"].update(readout_over)
    return document


fr.load_recipe, mr.load_recipe = _titans_recipe, _readout_recipe


class Captured(Exception):
    pass


captured = {}


def _intercept(build, fresh, paused, _options, inputs, _settings, counters=()):
    captured.update(build=build, fresh=fresh, paused=paused, inputs=inputs)
    raise Captured


ct._chronological = _intercept
campaign = extended_campaign(
    load_extensions(), load_campaign("configs/baselines/historical-masked-campaign-a.json")
)
section = {
    "titans": "titans_mac",
    "cm_core": "cm_v1",
    "readout": "mars_titan",
    "cm_readout": "cm_v1",
    "candidate": "episodic_gru",
}[args.family]
campaign[section]["candidates"] = {
    arm: value
    for arm, value in campaign[section]["candidates"].items()
    if arm == args.arm or (args.family == "cm_readout" and arm.startswith("cm_v1_core"))
}
if args.family == "cm_core":
    ct.CM_CORES, ct.CM_ARMS = (args.arm,), {}
if args.family == "cm_readout":
    ct.CM_CORES = ()
    ct.CM_ARMS = {args.arm: ct.CM_ARMS[args.arm]}
measure = {
    "titans": ct.measure_titans,
    "cm_core": ct.measure_cm_v1,
    "readout": ct.measure_mars_titan,
    "cm_readout": ct.measure_cm_v1,
    "candidate": ct.measure_candidate,
}[args.family]
started = time.perf_counter()
try:
    if args.family.startswith("cm"):
        measure(campaign, args.view, args.work, segments=8)
    else:
        measure(campaign, args.view, args.work)
except Captured:
    pass
else:
    raise SystemExit("No se interceptó el constructor del brazo")
setup_seconds = time.perf_counter() - started


class Capture(ct._NoStepOptimizer):
    def step(self):
        super().step()
        if args.capture is not None:
            self.records.append(
                [
                    None if v.grad is None else v.grad.detach().cpu().clone()
                    for v in self.parameters()
                ]
            )


class Budget:
    """Barrera tras cada paso: calentamiento, barrera entre trabajos y ventana medida."""

    def __init__(self, run):
        self.run, self.calls, self.marks, self.result = run, 0, [], None

    @property
    def requested(self):
        if self.result is not None:
            return True
        index, self.calls = self.calls, self.calls + 1
        torch.cuda.synchronize(0)
        if index == args.warmup and args.barrier is not None:
            args.barrier.mkdir(parents=True, exist_ok=True)
            (args.barrier / str(os.getpid())).touch()
            limit = time.monotonic() + 1800
            while len(list(args.barrier.iterdir())) < args.jobs:
                if time.monotonic() > limit:
                    raise SystemExit("La barrera entre trabajos no se completó")
                time.sleep(0.05)
        now = time.perf_counter()
        self.marks.append(
            (now, self.run.counters["observations"], self.run.counters.get("labels", 0), DATA[0])
        )
        if index == args.warmup + args.segments:
            self.result = True
            return True
        return False


guard = ct._forbid_steps()
context = titans.unfused_attention()
context.__enter__()
torch.cuda.reset_peak_memory_stats(0)
trainer = captured["build"](options)
trainer.optimizer.__class__ = Capture
trainer.optimizer.records = []
run = captured["fresh"](trainer)
original = trainer._labels


def _labels(run, *head, **kwargs):
    *rest, event = head
    kept = tuple(item for item in event.labels if (item[0], item[1]) in run.pending)
    return original(run, *rest, replace(event, labels=kept), **kwargs)


trainer._labels = _labels
# Tiempo dentro del generador de eventos: lectura, decodificación y bloques del índice.
DATA = [0.0]
_events = trainer.train.batched_events


def _timed_events(*a, **k):
    iterator = _events(*a, **k)
    while True:
        began = time.perf_counter()
        try:
            item = next(iterator)
        except StopIteration:
            DATA[0] += time.perf_counter() - began
            return
        DATA[0] += time.perf_counter() - began
        yield item


trainer.train.batched_events = _timed_events
REPLAY = None
if args.events is not None:
    import pickle

    import numpy as np

    with args.events.open("rb") as handle:
        REPLAY = pickle.load(handle)

    def _rows(raw, first, last):
        result = {}
        for key, value in raw.items():
            if isinstance(value, dict):
                result[key] = _rows(value, first, last)
            elif isinstance(value, (np.ndarray, list, tuple)):
                result[key] = value[first:last]
            else:
                result[key] = value
        return result

    def _replay_events(*, start_cursor=0, block_rows=256, **_):
        # Los eventos guardados son completos. Se trocean igual que `_BlockReader.blocks`.
        if start_cursor != REPLAY["start"]:
            raise SystemExit("Los eventos guardados empiezan en otro cursor")
        for event in REPLAY["events"]:
            chunks = []
            for raw in event.inputs:
                size = len(raw["market"])
                chunks.extend(
                    _rows(raw, first, min(first + block_rows, size))
                    for first in range(0, size, block_rows)
                )
            yield replace(event, inputs=tuple(chunks))

    _events = _replay_events
inputs = captured["inputs"]
start = int(args.start_fraction * len(inputs))
budget = Budget(run)
cursor = dict(epoch=0, phase="train", event=start, stage="start")
began = time.perf_counter()
truncated = False
try:
    trainer._train_pass(run, cursor, budget, ct._no_save)
except captured["paused"]:
    pass
except ValueError as error:
    # Los eventos guardados se agotan antes del cierre: se usan los tramos completos.
    if REPLAY is None or "sin su cierre" not in str(error):
        raise
    truncated = True
context.__exit__(None, None, None)
guard.remove()
assert trainer.optimizer.unchanged(), "La medición ha modificado parámetros"
marks = budget.marks[args.warmup :]
(t0, r0, l0, d0), (t1, r1, l1, d1) = marks[0], marks[-1]
gaps = [b[0] - a[0] for a, b in zip(marks, marks[1:], strict=False)]
rates = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(marks, marks[1:], strict=False)]


def process_mib():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        pid, used = (part.strip() for part in line.split(","))
        if pid == str(os.getpid()):
            return int(used)
    return None


ordered = sorted(rates)
result = dict(
    label=args.label,
    family=args.family,
    arm=args.arm,
    options=options,
    predictor=predictor_over,
    recipe=recipe_over,
    readout=readout_over,
    jobs=args.jobs,
    mps=bool(os.environ.get("CUDA_MPS_PIPE_DIRECTORY")),
    start_event=start,
    events=len(inputs),
    replay=None if args.events is None else str(args.events),
    truncated=truncated,
    max_event_inputs=max(inputs),
    segments=len(gaps),
    rows=r1 - r0,
    labels=l1 - l0,
    seconds=t1 - t0,
    rows_per_s=(r1 - r0) / (t1 - t0),
    data_seconds=d1 - d0,
    data_fraction=(d1 - d0) / (t1 - t0),
    segment_rate_median=ordered[len(ordered) // 2],
    segment_rate_min=ordered[0],
    segment_rate_max=ordered[-1],
    optimizer_calls=trainer.optimizer.calls,
    peak_allocated_mib=round(torch.cuda.max_memory_allocated(0) / 2**20),
    peak_reserved_mib=round(torch.cuda.max_memory_reserved(0) / 2**20),
    process_mib=process_mib(),
    peak_rss_mib=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    setup_seconds=round(setup_seconds, 1),
    pass_seconds=round(time.perf_counter() - began, 1),
    loadavg=os.getloadavg(),
    threads=os.environ.get("OMP_NUM_THREADS"),
    torch=torch.__version__,
    tf32_matmul=torch.backends.cuda.matmul.allow_tf32,
    tf32_cudnn=torch.backends.cudnn.allow_tf32,
    matmul_precision=torch.get_float32_matmul_precision(),
    code=str(Path(fr.__file__).resolve().parents[3]),
)
line = json.dumps(result)
print(line, flush=True)
if args.output is not None:
    with args.output.open("a") as handle:
        handle.write(line + "\n")
if args.capture is not None:
    names = (
        {id(p): n for n, p in trainer.__dict__.get("predictor", trainer).named_parameters()}
        if hasattr(trainer, "predictor")
        else {}
    )
    for owner in ("readout", "model"):
        if hasattr(trainer, owner) and hasattr(getattr(trainer, owner), "named_parameters"):
            try:
                found = dict(getattr(trainer, owner).named_parameters())
            except (TypeError, ValueError):
                # El modelo nativo de la candidata no da pares (nombre, tensor). Se usan posiciones.
                found = {}
            names.update({id(p): f"{owner}.{n}" for n, p in found.items()})
    pending = {f"{k[0]}|{k[1]}": float(getattr(v, "issued", v)) for k, v in run.pending.items()}
    torch.save(
        dict(
            grads=trainer.optimizer.records,
            names=[
                names.get(id(v), f"param{i}") for i, v in enumerate(trainer.optimizer.parameters())
            ],
            pending=pending,
            result=result,
        ),
        args.capture,
    )
sys.stdout.flush()
