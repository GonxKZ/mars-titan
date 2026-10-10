"""Recorridos de memoria del informe de perfilado, sin modelos ni pasos de optimizador.

Órdenes:

- `index`: construye los índices de observaciones de ajuste y validación de una vista, como
  `campaign_throughput._chronological_sources`.
- `read`: lee los últimos instantes del tramo de ajuste con la tubería de la campaña, con
  `benchmarks/campaign_pipeline.py chronological`.
- `validation`: carga la validación residente de XGBoost con sus claves (mercado y fecha de
  predicción) y la copia a `cuda:0` como `external_corpus._execute`, sin construir la matriz
  de entrenamiento ni ajustar rondas.
- `matrix`: construye la matriz cuantizada de ajuste de XGBoost con la configuración de la
  campaña (`build_external_matrix`, las dos pasadas del iterador y sus páginas en disco) y
  la libera, sin ajustar ninguna ronda. Con `--validation` lee antes de liberarla la
  validación residente, como el ajuste. Escribe las páginas en `TMPDIR` y registra también
  la VRAM del proceso.

Cada orden imprime un JSON con el tiempo, el pico de RSS del proceso, el pico del grupo de
control de `memslot` cuando es legible y el momento del pico según un muestreo de RSS cada
0,1 s, con su parte anónima por fase. Se ejecutan igual con y sin `memray run`, que
atribuye el pico por asignación.
"""

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TITANS_RECIPE = "configs/titans/chronological-training-historical-masked.json"
TABULAR = ROOT / "configs/baselines/tabular-historical-masked.json"


class Timeline:
    """RSS del proceso cada 0,1 s, su parte anónima y la marca de la fase en curso.

    La parte anónima (`RssAnon`) excluye las páginas de archivos proyectados en memoria, como
    las bibliotecas de CUDA, que el sistema puede reclamar.

    Con `device=True` lee además cada 0,5 s la memoria de GPU de este proceso según
    nvidia-smi, que incluye el contexto CUDA y la reserva de cada asignador. Si se asigna
    `extra`, una función sin argumentos, su resultado se guarda cada segundo junto al RSS en
    una traza para ver qué parte del crecimiento corresponde a cada estructura.
    """

    def __init__(self, device=False):
        self.samples, self.phase, self.stop = [], "start", threading.Event()
        self.anon_peaks = {}
        self.device, self.device_peaks = device, {}
        self.extra, self.trace = None, []
        self.began = time.perf_counter()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @staticmethod
    def rss():
        with open("/proc/self/statm") as handle:
            return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")

    @staticmethod
    def anon():
        with open("/proc/self/status") as handle:
            for line in handle:
                if line.startswith("RssAnon:"):
                    return int(line.split()[1]) * 1024
        raise ValueError("El núcleo no informa de RssAnon")

    @staticmethod
    def vram():
        """MiB de GPU de este proceso, o None si nvidia-smi no lo lista todavía."""
        output = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for line in output.splitlines():
            pid, _, used = line.partition(",")
            if pid.strip() == str(os.getpid()):
                return int(used)
        return None

    def _run(self):
        ticks = 0
        while not self.stop.wait(0.1):
            self.samples.append((time.perf_counter() - self.began, self.phase, self.rss()))
            anon = self.anon()
            self.anon_peaks[self.phase] = max(self.anon_peaks.get(self.phase, 0), anon)
            ticks += 1
            if self.extra is not None and ticks % 10 == 0:
                elapsed, phase, rss = self.samples[-1]
                self.trace.append(
                    dict(seconds=round(elapsed, 1), phase=phase, rss=rss, anon=anon, **self.extra())
                )
            if self.device and ticks % 5 == 0:
                used = self.vram()
                if used is not None:
                    phase = self.phase
                    self.device_peaks[phase] = max(self.device_peaks.get(phase, 0), used)

    def mark(self, phase):
        self.phase = phase

    def close(self):
        self.stop.set()
        self.thread.join()
        peak = max(self.samples, key=lambda item: item[2], default=(0.0, self.phase, 0))
        phases = {}
        for _, phase, value in self.samples:
            phases[phase] = max(phases.get(phase, 0), value)
        result = dict(
            sampled_peak_rss_bytes=peak[2],
            sampled_peak_seconds=round(peak[0], 2),
            sampled_peak_phase=peak[1],
            sampled_peak_by_phase=phases,
            sampled_anon_peak_by_phase=self.anon_peaks,
        )
        if self.device:
            result["device_peak_mib_by_phase"] = self.device_peaks
        if self.trace:
            result["trace"] = self.trace
        return result


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


def _features(data, config, presence):
    import numpy as np

    from mars_titan.training.tabular_corpus import _matrix

    first = next(data.batches(partition="train", batch_size=config["batch_size"], epoch=0, seed=0))
    return _matrix(first, np.float32, presence=presence).shape[1]


def validation(args, timeline):
    from mars_titan.data.input_policy import HISTORICAL_MASKED, masked_inputs

    config = json.loads(TABULAR.read_text())
    data = dataset(args.view)
    presence = masked_inputs(HISTORICAL_MASKED)
    return _resident(data, config, presence, _features(data, config, presence), timeline)


def _resident(data, config, presence, features, timeline):
    """Validación residente y su copia a `cuda:0`, como `external_corpus._execute`."""
    import cupy as cp
    import numpy as np

    from mars_titan.models.baselines.boosting_selection import ResidentValidation
    from mars_titan.training.external_corpus import _device_budget
    from mars_titan.training.tabular_corpus import _matrix

    batch_size = config["batch_size"]
    rows = data.manifest["counts"]["validation"]
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


def matrix(args, timeline):
    import numpy as np
    import pyarrow as pa

    from mars_titan.data.input_policy import HISTORICAL_MASKED, masked_inputs
    from mars_titan.models.baselines.external_boosting import (
        available_ram_bytes,
        build_external_matrix,
        external_cache_plan,
        free_disk_bytes,
    )
    from mars_titan.training.external_corpus import _TrainRows
    from mars_titan.training.tabular_corpus import _matrix

    config = json.loads(TABULAR.read_text())
    if args.max_bin not in config["bins"]:
        raise SystemExit(f"La campaña no declara max_bin={args.max_bin}")
    batch_size = config["batch_size"]
    data = dataset(args.view)
    presence = masked_inputs(HISTORICAL_MASKED)
    rows = data.manifest["counts"]["train"]
    features = _features(data, config, presence)
    recorded = _TrainRows(rows)
    timeline.extra = lambda: dict(
        reader_cached=data.cached_bytes,
        arrow_allocated=pa.total_allocated_bytes(),
        recorded_batches=len(recorded.parts or ()),
    )

    # La misma factoría que `external_corpus._execute`, con las claves de cada fila.
    def factory():
        recording = recorded.start()
        for batch in data.batches(partition="train", batch_size=batch_size, epoch=0, seed=0):
            if recording:
                recorded.add(batch)
            yield _matrix(batch, np.float32, presence=presence), batch["target"]
        if recording:
            recorded.finish()

    directory = Path(os.environ["TMPDIR"]) / f"matrix-{args.label}"
    construction = dict(
        expected_rows=rows,
        max_bin=args.max_bin,
        max_batch_bytes=config["max_batch_bytes"],
        max_host_cache_bytes=config["max_host_cache_bytes"],
        on_host=config["on_host"],
        max_disk_cache_bytes=config["max_disk_cache_bytes"],
    )
    # El plan que `run_external_reference` contrasta antes de construir, sin validación.
    plan = external_cache_plan(
        rows=rows,
        features=features,
        max_bin=args.max_bin,
        on_host=config["on_host"],
        max_host_cache_bytes=config["max_host_cache_bytes"],
        max_disk_cache_bytes=config["max_disk_cache_bytes"],
        available_ram=available_ram_bytes(),
        free_disk=free_disk_bytes(directory),
    )
    timeline.mark("matrix_build")
    began = time.perf_counter()
    built = build_external_matrix(factory, directory, **construction)
    build_seconds = time.perf_counter() - began
    timeline.mark("matrix_alive")
    disk = built.disk_bytes()
    retained = Timeline.rss()
    keys = sum(value.nbytes for value in (recorded.china, recorded.moments, recorded.target))
    reader_cached = data.cached_bytes
    audit = dict(built.audit)
    # Como en `_execute`, la validación residente se construye con la matriz viva.
    resident = _resident(data, config, presence, features, timeline) if args.validation else None
    timeline.mark("matrix_release")
    built.close()
    return dict(
        rows=rows,
        features=features,
        construction=construction,
        plan={
            key: plan[key]
            for key in (
                "cache_location",
                "host_cache_bytes_estimate",
                "disk_cache_bytes_estimate",
                "disk_cache_bytes_global_bins_bound",
                "max_disk_cache_bytes",
            )
        },
        disk_bytes=disk,
        disk_bytes_per_row=disk / rows,
        build_seconds=build_seconds,
        retained_rss_bytes=retained,
        row_keys_bytes=keys,
        reader_cached_bytes=reader_cached,
        validation=resident,
        audit={key: audit[key] for key in ("cache_location", "completed_pass_rows")},
        max_input_batch_bytes=audit["max_input_batch_bytes"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "read", "validation", "matrix"):
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
    commands.choices["matrix"].add_argument("--max-bin", type=int, default=256)
    commands.choices["matrix"].add_argument("--validation", action="store_true")
    args = parser.parse_args()
    import pyarrow as pa

    timeline = Timeline(device=args.command in {"validation", "matrix"})
    began = time.perf_counter()
    commands_ = {"index": index, "read": read, "validation": validation, "matrix": matrix}
    result = commands_[args.command](args, timeline)
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
