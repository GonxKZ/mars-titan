"""Selección anclada del banco con y sin las opciones que conservan los bits.

Mide `select_anchored_medoids` con la forma de las llamadas del recorrido real de los
lectores de MARS-TITAN y CM-v1 (1.024 episodios retenidos y 4.040 nuevos, 1.016 centros
fijos y 16 candidatos, 64 coordenadas FP32) sobre dos geometrías sintéticas. Compara la
ruta de referencia con `bounded_background` y `reuse_distances`, exige los mismos campos
salvo la memoria declarada y las propias opciones, y registra las medianas. La poda de la
cota depende de la geometría, así que estas cifras no sustituyen a la medida con claves
reales del informe. Sin banco, sin modelo y sin GPU.

Uso: `PYTHONPATH=src python benchmarks/anchored_medoid_options.py salida.json`
"""

import dataclasses
import json
import os
import platform
import statistics
import sys
import time

import numpy as np

from mars_titan.cm.anchored_medoids import select_anchored_medoids

RETAINED, INCOMING, FIXED, CANDIDATES, DIMENSIONS = 1024, 4040, 1016, 16, 64
OPTIONS = dict(reuse_distances=True, bounded_background=True)
CHANGED = {"reused_distances", "bounded_background", "estimated_peak_bytes"}
REPETITIONS = 5


def geometry(kind, rng):
    """Puntos FP32: una normal isótropa o 32 grupos separados."""
    count = RETAINED + INCOMING
    if kind == "isotropic":
        return rng.standard_normal((count, DIMENSIONS)).astype(np.float32)
    centers = 4 * rng.standard_normal((32, DIMENSIONS))
    labels = rng.integers(0, 32, count)
    return (centers[labels] + 0.5 * rng.standard_normal((count, DIMENSIONS))).astype(np.float32)


def problem(kind, seed=617):
    """Llamada como la de la política anclada: frontera de 8 retenidos y 8 nuevos al azar."""
    rng = np.random.default_rng(seed)
    points = geometry(kind, rng)
    ids = [f"{index:020d}" for index in range(len(points))]
    fixed = ids[8:RETAINED]
    order = rng.permutation(INCOMING) + RETAINED
    candidates = sorted(ids[:8] + [ids[index] for index in order[: CANDIDATES - 8]])
    return points, ids, fixed, candidates


def same_fields(reference, other):
    return all(
        getattr(reference, field.name) == getattr(other, field.name)
        for field in dataclasses.fields(reference)
        if field.name not in CHANGED
    ) and repr(reference.objective) == repr(other.objective)


def measure(kind):
    points, ids, fixed, candidates = problem(kind)
    variants = {"reference": {}, "bounded_and_reused": OPTIONS}
    results, seconds = {}, {name: [] for name in variants}
    for name, options in variants.items():
        results[name] = select_anchored_medoids(points, ids, fixed, candidates, RETAINED, **options)
    for repetition in range(REPETITIONS):
        # Se alterna el orden para repartir entre las dos variantes la deriva de la máquina.
        order = list(variants) if repetition % 2 == 0 else list(reversed(variants))
        for name in order:
            began = time.perf_counter()
            result = select_anchored_medoids(
                points, ids, fixed, candidates, RETAINED, **variants[name]
            )
            seconds[name].append(time.perf_counter() - began)
            if not same_fields(results[name], result):
                raise RuntimeError("Dos llamadas iguales devolvieron selecciones distintas")
    reference, optimized = results["reference"], results["bounded_and_reused"]
    medians = {name: statistics.median(values) for name, values in seconds.items()}
    return dict(
        geometry=kind,
        clients=len(points) - len(fixed),
        fixed=len(fixed),
        candidates=len(candidates),
        same_fields_and_objective_bits=same_fields(reference, optimized),
        applied=dict(
            reused_distances=optimized.reused_distances,
            bounded_background=optimized.bounded_background,
        ),
        swaps=reference.swaps,
        distance_pairs=reference.distance_pairs,
        estimated_peak_bytes=dict(
            reference=reference.estimated_peak_bytes,
            bounded_and_reused=optimized.estimated_peak_bytes,
        ),
        seconds=seconds,
        median_seconds=medians,
        speedup=medians["reference"] / medians["bounded_and_reused"],
    )


def main(output):
    report = dict(
        shape=dict(
            retained=RETAINED,
            incoming=INCOMING,
            fixed=FIXED,
            candidates=CANDIDATES,
            dimensions=DIMENSIONS,
        ),
        repetitions=REPETITIONS,
        warmup_calls_per_variant=1,
        measurements=[measure(kind) for kind in ("isotropic", "clustered")],
        versions=dict(python=platform.python_version(), numpy=np.__version__),
        threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        load_average=list(os.getloadavg()),
        limits=[
            "Geometrías sintéticas. La poda de la cota depende de la geometría de las claves.",
            "No incluye banco nativo, codec, snapshots ni modelo.",
            "Otras aplicaciones siguieron activas y no se mide energía.",
        ],
    )
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    if not all(row["same_fields_and_objective_bits"] for row in report["measurements"]):
        raise SystemExit("Las opciones cambiaron la selección")


if __name__ == "__main__":
    main(sys.argv[1])
