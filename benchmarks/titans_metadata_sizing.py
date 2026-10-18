"""Medir el recuento de metadatos frente al JSON completo, sin ajustar parámetros."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.titans.config import canonical
from tests.models.titans.test_financial_adapter import setup


def measure(repeats=200):
    """Alternar ambos recuentos sobre identidades ficticias de tamaño declarado."""
    if type(repeats) is not int or not 20 <= repeats <= 10000:
        raise ValueError("La medida necesita entre 20 y 10.000 repeticiones")
    model, _ = setup()
    results = []
    for size in (4, 128, 1024):
        flows = tuple(f"US/A{i}" for i in range(size))
        moments = (1_609_459_200_000_000,) * size
        ids = tuple(f"{flow}/{at}" for flow, at in zip(flows, moments, strict=True))
        times = dict(reference=[], cached=[])
        functions = dict(
            reference=lambda flows=flows, ids=ids, moments=moments: len(
                canonical(model._metadata(flows, ids, moments)).encode()
            ),
            cached=lambda flows=flows, ids=ids, moments=moments: model._metadata_bytes(
                flows, ids, moments
            ),
        )
        expected = functions["reference"]()
        for index in range(20 + repeats):
            for name in times if index % 2 else reversed(times):
                started = time.perf_counter_ns()
                actual = functions[name]()
                elapsed = (time.perf_counter_ns() - started) / 1e9
                if actual != expected:
                    raise AssertionError("El recuento no conserva los bytes del JSON completo")
                if index >= 20:
                    times[name].append(elapsed)
        results.append(
            dict(
                flows=size,
                metadata_bytes=expected,
                seconds=times,
                percentiles_seconds={
                    name: dict(
                        zip(
                            ("p50", "p95", "p99"),
                            np.quantile(values, (0.5, 0.95, 0.99)),
                            strict=True,
                        )
                    )
                    for name, values in times.items()
                },
            )
        )
    return dict(
        kind="titans_metadata_sizing",
        origin="generated_identity_fixture",
        optimizer_steps=0,
        device="cpu",
        warmup=20,
        repeats=repeats,
        exact_bytes=True,
        script_sha256=sha256(Path(__file__)),
        results=results,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=200)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("La medida necesita un destino nuevo")
    result = measure(args.repeats)
    atomic_json(args.output, result)
    print(json.dumps(dict(exact_bytes=result["exact_bytes"], optimizer_steps=0)))


if __name__ == "__main__":
    main()
