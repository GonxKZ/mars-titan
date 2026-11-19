"""Medir en CPU el coste de calcular la ruta de cada fuente de rol de B3 (#750), sin ajustar nada.

Las ventanas y la presencia son sintéticas con semilla fija y solo fijan las formas reales:
ventanas [n, 64, 6] en FP32 de un mercado, con cierres de paseo aleatorio y todas las sesiones
presentes, y los cinco bits de modalidad de cada fila. Cada fuente calcula las rutas de un
evento completo y las repeticiones alternan las fuentes con orden rotatorio. El ciclo de
claves, lectura y escritura no depende de la fuente y lo mide `role_binding_cost.py`.
"""

import argparse
import hashlib
import os
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import scipy

from benchmarks.regime_routing_cost import AT, windows
from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.regimes import SLOTS
from mars_titan.memory.role_binding import SOURCES, RoleBindingConfig

SIZES = (64, 256, 1024)
CODE = (
    "src/mars_titan/memory/role_sources.py",
    "src/mars_titan/memory/regimes.py",
    "src/mars_titan/memory/role_binding.py",
    "src/mars_titan/data/correlation_regions.py",
    "benchmarks/role_sources_cost.py",
)


def presence(rows):
    """Precios, gráficos y macro presentes, noticias en filas pares y fundamentales en 2 de 3."""
    values = np.ones((rows, len(MODALITIES)), dtype=np.bool_)
    values[:, MODALITIES.index("news")] = np.arange(rows) % 2 == 0
    values[:, MODALITIES.index("fundamentals")] = np.arange(rows) % 3 != 0
    return values


def measure(output, *, sizes=SIZES, warmup=3, repetitions=20):
    if (
        not sizes
        or len(set(sizes)) != len(sizes)
        or any(type(rows) is not int or not 1 <= rows <= 4096 for rows in sizes)
        or type(warmup) is not int
        or not 0 <= warmup <= 10
        or type(repetitions) is not int
        or not 1 <= repetitions <= 100
    ):
        raise ValueError("Los tamaños o las repeticiones exceden el presupuesto de la medida")
    if Path(output).exists():
        raise FileExistsError("La medida necesita un recibo nuevo")
    started, load = time.perf_counter(), os.getloadavg()
    names = list(SOURCES)
    rules = {
        name: MatureCorrection(
            AssociativeMemoryConfig("proximal"), role_binding=RoleBindingConfig(source=name)
        ).routing
        for name in names
    }
    cases = {}
    for rows in sizes:
        prices = windows(rows, np.random.default_rng(20261030 + rows))
        flags = presence(rows)
        flows = [f"US/A{index:05d}" for index in range(rows)]
        samples = {name: [] for name in names}
        for repetition in range(warmup + repetitions):
            shift = repetition % len(names)
            for name in names[shift:] + names[:shift]:
                begin = time.perf_counter()
                rules[name].routes(prices, flows, AT, flags)
                elapsed = time.perf_counter() - begin
                if repetition >= warmup:
                    samples[name].append(elapsed)
        for name in names:
            ordered = sorted(samples[name])
            routes, _ = rules[name].routes(prices, flows, AT, flags)
            cases[f"{name}/{rows}"] = dict(
                p50_ms=statistics.median(ordered) * 1e3,
                p95_ms=ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))] * 1e3,
                min_ms=ordered[0] * 1e3,
                samples_ms=[value * 1e3 for value in samples[name]],
                occupancy=np.bincount(routes, minlength=SLOTS).tolist(),
            )
    receipt = dict(
        schema_version=1,
        kind="role_source_cost",
        issue=750,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        sources_sha256={path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in CODE},
        data="sintéticos con semilla fija, solo para fijar formas. Nada se ajusta ni evalúa",
        rules={name: rule.identity() for name, rule in rules.items()},
        shapes=dict(windows=[64, 6], presence=len(MODALITIES), market="US", rows=list(sizes)),
        warmup=warmup,
        repetitions=repetitions,
        optimizer_steps=0,
        training_runs=0,
        environment=dict(
            cpu=cpu_name(),
            numpy=np.__version__,
            scipy=scipy.__version__,
            python=platform.python_version(),
            omp_num_threads=os.environ.get("OMP_NUM_THREADS"),
            load_average_at_start=load,
            load_average_at_end=os.getloadavg(),
        ),
        elapsed_seconds=time.perf_counter() - started,
        not_measured=[
            "GPU, porque las rutas se calculan en CPU",
            "el ciclo de claves, lectura y escritura, que no depende de la fuente",
            "energía, sin instrumento",
        ],
        cases=cases,
    )
    atomic_json(output, receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=list(SIZES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    args = parser.parse_args()
    measure(args.output, sizes=tuple(args.sizes), warmup=args.warmup, repetitions=args.repetitions)


if __name__ == "__main__":
    main()
