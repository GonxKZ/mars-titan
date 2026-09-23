"""Comparar preparación de etiquetas en procesos separados y salidas nuevas."""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.corpus_targets import _json, prepare_corpus_targets

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("manifest", type=Path)
parser.add_argument("prepared", type=Path)
parser.add_argument("output", type=Path)
parser.add_argument("--repetitions", type=int, default=3, choices=range(1, 11))
parser.add_argument("--worker", choices=("reference", "numpy"), help=argparse.SUPPRESS)
args = parser.parse_args()
if sys.platform != "linux":
    raise ValueError("Esta medición de memoria requiere /proc de Linux")
if args.worker:
    start = time.perf_counter()
    receipt = prepare_corpus_targets(args.manifest, args.prepared, args.output, backend=args.worker)
    elapsed = time.perf_counter() - start
    peak = next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/self/status").read_text().splitlines()
        if line.startswith("VmHWM:")
    )
    print(
        json.dumps(
            dict(
                backend=args.worker,
                compute_seconds=elapsed,
                process_peak_rss_bytes=peak,
                assets=len(receipt["assets"]),
                counts=receipt["counts"],
                reused_assets=receipt["reused_assets"],
                label_hashes={
                    a["market"] + "/" + a["symbol"]: a["labels_sha256"] for a in receipt["assets"]
                },
            )
        )
    )
else:
    metadata = _json(args.manifest)
    for source in (Path("dataset"), args.prepared, Path(metadata["samples_root"])):
        outside_source(source, args.output)
        outside_source(args.output, source)
    args.output.mkdir(parents=True, exist_ok=False)
    measurements, population = [], None
    for repeat in range(args.repetitions):
        order = ("reference", "numpy") if repeat % 2 == 0 else ("numpy", "reference")
        for backend in order:
            output = args.output / f"{backend}-{repeat}"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                str(args.manifest),
                str(args.prepared),
                str(output),
                "--worker",
                backend,
            ]
            start = time.perf_counter()
            process = subprocess.run(command, check=True, text=True, capture_output=True)
            measurement = json.loads(process.stdout)
            measurement.update(repeat=repeat, process_seconds=time.perf_counter() - start)
            if measurement["reused_assets"]:
                raise ValueError("La medición ha reutilizado etiquetas existentes")
            if population is None:
                population = measurement["label_hashes"]
            elif population != measurement["label_hashes"]:
                raise ValueError("Las etiquetas no coinciden exactamente entre motores")
            measurements.append(measurement)
            print(
                json.dumps({k: v for k, v in measurement.items() if k != "label_hashes"}),
                flush=True,
            )
    result = dict(
        source_manifest_sha256=sha256(args.manifest),
        benchmark_sha256=sha256(Path(__file__)),
        code_sha256={
            name: sha256(Path(name))
            for name in (
                "src/mars_titan/data/budget_targets.py",
                "src/mars_titan/data/residual_arrays.py",
                "src/mars_titan/training/corpus_targets.py",
            )
        },
        python=platform.python_version(),
        machine=platform.machine(),
        threads={k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
        assets=len(population),
        exact_labels_equal=True,
        measurements=measurements,
        limits=[
            "Procesos separados. La caché del sistema operativo no se vacía.",
            "El tiempo de proceso incluye arranque e importaciones.",
            "El tiempo interno incluye lectura, cálculo, huellas y escritura.",
            "No se ejecuta trabajo GPU ni se mide energía en esta comparación.",
        ],
    )
    atomic_json(args.output / "measurement.json", result)
