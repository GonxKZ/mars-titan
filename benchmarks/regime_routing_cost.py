"""Medir en CPU el coste del enrutamiento por régimen de B6 frente a las claves sin enrutar.

Las ventanas, las entradas del codec y los valores son sintéticos con semilla fija y solo
fijan las formas reales: ventanas [n, 64, 6] en FP32 de un mercado, entradas del codec de 64
coordenadas y cohortes de n etiquetas. No se ajusta ni se evalúa nada. Para cada tamaño se
mide el cálculo de la ruta, las claves, la lectura de A y la escritura proximal de la cohorte
con cada clave, con tres calentamientos y veinte repeticiones. El informe compara la suma por
evento con el tiempo del núcleo a su caudal medido, que se pasa como argumento.
"""

import argparse
import hashlib
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.associative_memory import (
    KEY_SIZES,
    AssociativeMemory,
    AssociativeMemoryConfig,
    MatureCorrection,
)
from mars_titan.memory.regimes import REGIME_RULE, RegimeRule

SIZES = (64, 128, 1024, 4096)
KEYS = ("constant", "regime", "codec", "codec_by_regime")
AT = 1_615_000_000_000_000


def windows(rows, generator):
    """Ventanas con cierres de paseo aleatorio, todas las sesiones presentes y un mercado."""
    steps = generator.normal(0.0, 0.01, size=(rows, 63))
    closes = np.concatenate([np.zeros((rows, 1)), np.cumsum(steps, axis=1)], axis=1)
    values = np.repeat(closes[:, :, None], 6, axis=2)
    values[:, :, 4], values[:, :, 5] = 0.1, 1.0
    return values.astype(np.float32)


def timed(function, warmup, repeats):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    ordered = sorted(samples)
    return dict(
        p50_ms=statistics.median(samples) * 1e3,
        p95_ms=ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))] * 1e3,
        min_ms=ordered[0] * 1e3,
        repeats=repeats,
    )


def correction(key):
    config = AssociativeMemoryConfig(
        "proximal", key_size=KEY_SIZES[key], rate=0.25, forgetting=0.01
    )
    return MatureCorrection(config, key=key)


def measure(rows, key, generator, warmup, repeats):
    """Ruta, claves, lectura y escritura de una cohorte de `rows` con la clave `key`."""
    corrections = correction(key)
    prices = windows(rows, generator)
    flows = [f"US/A{index:05d}" for index in range(rows)]
    inputs = torch.from_numpy(generator.normal(size=(rows, 64)).astype(np.float32))
    routes = None
    entry = {}
    if corrections.routing is not None:
        entry["route"] = timed(lambda: corrections.routes(prices, flows, AT), warmup, repeats)
        routes = corrections.routes(prices, flows, AT)[0]
    keys = corrections.keys(inputs, routes)
    entry["keys"] = timed(lambda: corrections.keys(inputs, routes), warmup, repeats)
    memory = AssociativeMemory(corrections.memory)
    values = (0.01 * generator.normal(size=rows)).tolist()

    def feedback(start):
        return corrections.feedback(
            ids=list(range(start, start + rows)),
            decision_at=[AT] * rows,
            available_at=[AT + 1] * rows,
            keys=keys,
            values=values,
        )

    # Un estado ya escrito evita medir la primera escritura desde cero como caso típico.
    memory = memory.write(feedback(1), cutoff=AT + 1)
    later = feedback(rows + 1)
    later = type(later)(
        ids=later.ids,
        decision_at=later.decision_at + 10,
        available_at=later.available_at + 10,
        keys=later.keys,
        values=later.values,
        weights=later.weights,
    )
    entry["read"] = timed(lambda: memory.read(keys), warmup, repeats)
    entry["write"] = timed(lambda: memory.write(later, cutoff=AT + 11), warmup, repeats)
    entry["event_p50_ms"] = sum(entry[name]["p50_ms"] for name in entry)
    entry["state_bytes"] = memory.matrix.nbytes
    entry["key_bytes"] = keys.nbytes
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--core-rows-per-second",
        type=float,
        required=True,
        help="Caudal de inferencia del núcleo con el que se compara el coste por evento",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "2")))
    generator = np.random.default_rng(20261010)
    cases = {}
    for rows in SIZES:
        core_ms = 1e3 * rows / args.core_rows_per_second
        for key in KEYS:
            entry = measure(rows, key, generator, args.warmup, args.repeats)
            entry["share_of_core_event"] = entry["event_p50_ms"] / core_ms
            cases[f"{key}/{rows}"] = entry
        for routed, plain in (("regime", "constant"), ("codec_by_regime", "codec")):
            extra = (
                cases[f"{routed}/{rows}"]["event_p50_ms"] - cases[f"{plain}/{rows}"]["event_p50_ms"]
            )
            cases[f"{routed}_minus_{plain}/{rows}"] = dict(
                event_p50_ms=extra, share_of_core_event=extra / core_ms
            )
    receipt = dict(
        schema_version=1,
        kind="regime_routing_cost",
        issue=20,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        sources_sha256={
            path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for path in (
                "src/mars_titan/memory/regimes.py",
                "src/mars_titan/memory/associative_memory.py",
                "benchmarks/regime_routing_cost.py",
            )
        },
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        rule=RegimeRule(REGIME_RULE).identity(),
        shapes=dict(windows=[64, 6], codec=64, value_size=1, cohorts=list(SIZES)),
        core_rows_per_second=args.core_rows_per_second,
        environment=dict(
            cpu=cpu_name(),
            numpy=np.__version__,
            torch=torch.__version__,
            torch_threads=torch.get_num_threads(),
            python=platform.python_version(),
            load_average_at_start=os.getloadavg(),
        ),
        not_measured=[
            "GPU, porque la ruta y A trabajan en CPU y FP64",
            "energía, sin instrumento",
            "el caudal del núcleo, que procede de la medida indicada",
        ],
        cases=cases,
    )
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=1) + "\n")
    for name, entry in cases.items():
        print(name, round(entry["event_p50_ms"], 3), round(entry["share_of_core_event"], 4))


if __name__ == "__main__":
    main()
