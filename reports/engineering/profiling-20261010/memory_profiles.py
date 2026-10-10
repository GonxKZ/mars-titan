"""Recorridos de memoria del informe de perfilado, sin modelos ni pasos de optimizador.

Órdenes:

- `index`: construye los índices de observaciones de ajuste y validación de una vista, como
  `campaign_throughput._chronological_sources`.
- `read`: lee los últimos instantes del tramo de ajuste con la tubería de la campaña, con
  `benchmarks/campaign_pipeline.py chronological`.
- `validation`: carga la validación residente de XGBoost con sus claves (mercado y fecha de
  predicción) y la copia a `cuda:0` como `external_corpus._execute`, sin construir la matriz
  de entrenamiento ni ajustar rondas.

Cada orden imprime un JSON con el tiempo, el pico de RSS del proceso, el pico del grupo de
control de `memslot` cuando es legible y el momento del pico según un muestreo de RSS cada
0,1 s. Se ejecutan igual con y sin `memray run`, que atribuye el pico por asignación.
"""

import argparse
import json
import math
import os
import resource
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TITANS_RECIPE = "configs/titans/chronological-training-historical-masked.json"
TABULAR = ROOT / "configs/baselines/tabular-historical-masked.json"


class Timeline:
    """RSS del proceso cada 0,1 s y la marca de la fase en curso."""

    def __init__(self):
        self.samples, self.phase, self.stop = [], "start", threading.Event()
        self.began = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def rss():
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    def _run(self):
        while not self.stop.wait(0.1):
            self.samples.append((time.perf_counter() - self.began, self.phase, self.rss()))

    def mark(self, phase):
        self.phase = phase

    def close(self):
        self.stop.set()
        self.thread.join()
        peak = max(self.samples, key=lambda item: item[2], default=(0.0, self.phase, 0))
        phases = {}
        for _, phase, value in self.samples:
            phases[phase] = max(phases.get(phase, 0), value)
        return dict(
            sampled_peak_rss_bytes=peak[2],
            sampled_peak_seconds=round(peak[0], 2),
            sampled_peak_phase=peak[1],
            sampled_peak_by_phase=phases,
        )


def cgroup_peak():
    try:
        group = Path("/proc/self/cgroup").read_text().strip().split(":", 2)[2]
        return int(Path(f"/sys/fs/cgroup{group}/memory.peak").read_text())
    except (OSError, ValueError, IndexError):
        return None


def dataset(view):
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.training.corpus_inputs import CorpusDataset

    return CorpusDataset(view, input_policy=HISTORICAL_MASKED)


def index(args, timeline):
    from mars_titan.training import titans_walk_forward as titans
    from mars_titan.training.campaign_throughput import _view_fold
    from mars_titan.training.financial_run import load_recipe

    _, document = load_recipe(ROOT / TITANS_RECIPE)
    data = dataset(args.view)
    phases = titans.window_phases(
        _view_fold(data), titans.walk_forward_options(document)["warmup_months"]
    )
    report = {}
    for name in ("train", "validation"):
        timeline.mark(f"index_{name}")
        began = time.perf_counter()
        source = titans._sources(data, {name: phases[name]}, args.index)[name]
        report[name] = dict(
            seconds=time.perf_counter() - began, events=len(source.metadata["groups"])
        )
    return report


def read(args, timeline):
    sys.path.insert(0, str(ROOT / "benchmarks"))
    import campaign_pipeline

    timeline.mark("read")
    return campaign_pipeline.chronological(args)


def validation(args, timeline):
    import cupy as cp
    import numpy as np

    from mars_titan.data.input_policy import HISTORICAL_MASKED, masked_inputs
    from mars_titan.models.baselines.boosting_selection import ResidentValidation
    from mars_titan.training.external_corpus import _device_budget
    from mars_titan.training.tabular_corpus import _matrix

    config = json.loads(TABULAR.read_text())
    batch_size = config["batch_size"]
    data = dataset(args.view)
    presence = masked_inputs(HISTORICAL_MASKED)
    rows = data.manifest["counts"]["validation"]
    first = next(data.batches(partition="train", batch_size=batch_size, epoch=0, seed=0))
    features = _matrix(first, np.float32, presence=presence).shape[1]
    del first
    # La misma declaración que `run_external_reference` pasa al plan de memoria.
    declared = min(
        config["max_validation_cache_bytes"],
        rows * (features * 4 + 24) + 4096 * math.ceil(rows / batch_size),
    )

    def factory():
        for batch in data.batches(partition="validation", batch_size=batch_size, epoch=0, seed=0):
            yield (
                _matrix(batch, np.float32, presence=presence),
                batch["target"],
                batch["market"],
                batch["prediction_at"],
            )

    timeline.mark("validation_read")
    began = time.perf_counter()
    resident = ResidentValidation(
        factory, expected_rows=rows, max_bytes=config["max_validation_cache_bytes"]
    )
    read_seconds = time.perf_counter() - began
    timeline.mark("validation_place")
    began = time.perf_counter()
    placed = resident.place(_device_budget(cp))
    cp.cuda.Device(0).synchronize()
    place_seconds = time.perf_counter() - began
    parts = {
        name: sum(block[index].nbytes for block in resident.blocks)
        for index, name in enumerate(("values", "target", "market", "prediction_at"))
    }
    dtypes = {
        name: str(resident.blocks[0][index].dtype)
        for index, name in enumerate(("values", "target", "market", "prediction_at"))
    }
    blocks = len(resident.blocks)
    timeline.mark("validation_release")
    resident.release_device()
    resident.release()
    return dict(
        rows=rows,
        features=features,
        batch_size=batch_size,
        blocks=blocks,
        resident_bytes=resident.bytes,
        resident_parts_bytes=parts,
        resident_dtypes=dtypes,
        declared_bytes=declared,
        device_bytes=placed,
        read_seconds=read_seconds,
        place_seconds=place_seconds,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "read", "validation"):
        command = commands.add_parser(name)
        command.add_argument("view", type=Path)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--label", required=True)
    for name in ("index", "read"):
        commands.choices[name].add_argument("--index", type=Path, required=True)
    reader = commands.choices["read"]
    reader.add_argument("--events", type=int, default=100)
    reader.add_argument("--start", type=int, default=-100)
    reader.add_argument("--warm", type=int, default=10)
    reader.add_argument("--block-rows", type=int, default=1024)
    reader.add_argument("--cache-mib", type=int, default=4096)
    reader.add_argument("--lru", action="store_true")
    args = parser.parse_args()
    import pyarrow as pa

    timeline = Timeline()
    began = time.perf_counter()
    result = {"index": index, "read": read, "validation": validation}[args.command](args, timeline)
    seconds = time.perf_counter() - began
    report = dict(
        command=args.command,
        result=result,
        seconds=seconds,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cgroup_memory_peak_bytes=cgroup_peak(),
        arrow_memory_pool=pa.default_memory_pool().backend_name,
        label=args.label,
        load_average=os.getloadavg(),
        pipeline={
            key: os.environ.get(key)
            for key in (
                "MARS_TITAN_DECODE_WORKERS",
                "MARS_TITAN_PREFETCH_BATCHES",
                "MARS_TITAN_GROUP_CACHE_MIB",
            )
        },
        **timeline.close(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("a") as handle:
        handle.write(json.dumps(report, default=str) + "\n")
    print(json.dumps(report, default=str))


if __name__ == "__main__":
    main()
