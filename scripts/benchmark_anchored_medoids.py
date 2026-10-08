"""Compara el selector público con fondo NumPy o SciPy, sin banco ni modelo."""

import hashlib
import json
import math
import os
import platform
import resource
import statistics
import time
import tracemalloc

import numpy as np
import scipy

from mars_titan.cm.anchored_medoids import select_anchored_medoids


def benchmark():
    """Medir conversión, validación, fondo y selección con la misma geometría FP32."""
    rng = np.random.default_rng(617)
    reports = []
    for incoming in (128, 512, 2048):
        points = rng.normal(size=(8192 + incoming, 64)).astype(np.float32)
        ids = tuple(f"{index:06d}" for index in range(len(points)))
        fixed, candidates = ids[:8184], ids[8184:8200]
        samples = {name: [] for name in ("numpy", "scipy_cdist_fp32")}
        results = {}
        for backend in samples:
            select_anchored_medoids(
                points, ids, fixed, candidates, 8192, background_backend=backend, max_swaps=1
            )
        for repetition in range(3):
            order = tuple(samples) if repetition % 2 == 0 else tuple(reversed(samples))
            for backend in order:
                start = time.perf_counter()
                result = select_anchored_medoids(
                    points, ids, fixed, candidates, 8192, background_backend=backend, max_swaps=1
                )
                samples[backend].append(time.perf_counter() - start)
                results[backend] = result
            reference, candidate = (results[name] for name in samples)
            if reference.retained_ids != candidate.retained_ids or not math.isclose(
                reference.objective, candidate.objective, rel_tol=1e-14, abs_tol=0
            ):
                raise ArithmeticError("El backend cambia los representantes o el objetivo")
        peaks = {}
        if incoming == 512:
            for backend in samples:
                tracemalloc.start()
                result = select_anchored_medoids(
                    points, ids, fixed, candidates, 8192, background_backend=backend, max_swaps=1
                )
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                if peak > result.estimated_peak_bytes:
                    raise RuntimeError("El pico rastreado supera la estimación de buffers")
                peaks[backend] = peak
        medians = {name: statistics.median(values) for name, values in samples.items()}
        reports.append(
            {
                "current_clients": len(points),
                "new_clients": incoming,
                "capacity": 8192,
                "fixed_centers": 8184,
                "variable_candidates": 16,
                "dimensions": 64,
                "coordinate_dtype": points.dtype.str,
                "distance_dtype": "float64",
                "warmup_calls_per_backend": 1,
                "seconds": samples,
                "median_seconds": medians,
                "observed_ratio": medians["numpy"] / medians["scipy_cdist_fp32"],
                "objective": result.objective,
                "variable_ids": result.variable_ids,
                "retained_ids_sha256": hashlib.sha256(
                    json.dumps(result.retained_ids).encode()
                ).hexdigest(),
                "background_pairs": result.background_pairs,
                "variable_pairs": result.variable_pairs,
                "status": result.status,
                "estimated_peak_bytes": result.estimated_peak_bytes,
                "tracemalloc_peak_bytes": peaks,
                "coordinate_input_bytes": points.nbytes,
            }
        )
    return {
        "reports": reports,
        "device": "cpu",
        "seed": 617,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "threads": {
            name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")
        },
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "limits": [
            "Incluye validación, IDs, conversión, fondo y selección mediante la API pública.",
            "No incluye codec, elegibilidad temporal, banco, modelo, snapshot o persistencia.",
            "La comparación exige mismos representantes, rtol=1e-14 y atol=0 para el objetivo.",
            "No se controlan otras aplicaciones ni se mide GPU, energía o coste económico.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), ensure_ascii=False, indent=2))
