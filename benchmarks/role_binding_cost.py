"""Coste técnico CPU de B3 y sus controles sobre las mismas claves de un fixture.

Incluye la construcción de claves, lectura y escritura madura. Excluye codificación,
predictor, cálculo del régimen, almacenamiento y recuperación. No mide error predictivo.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from benchmarks.area_memory_cost import timed
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.training.mars_titan_correction import _code, arm_components


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("El recibo de coste debe ser nuevo")
    if not 0 <= args.warmup <= 10 or not 1 <= args.repeats <= 100:
        raise ValueError("El calentamiento o las repeticiones exceden sus límites")
    if torch.cuda.is_initialized():
        raise RuntimeError("La medida del corrector CPU no utiliza CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    path = Path("configs/experiments/role-bound-keys-v1.json")
    document = json.loads(path.read_text())
    declaration = load_declaration()
    start, load = time.perf_counter(), os.getloadavg()
    cases = {}
    for name, arm in document["arms"].items():
        _, correction = check_components(
            declaration, arm_components(arm["components"], dict(rate=0.05, forgetting=0.01))
        )
        for rows in (64, 256, 1024):
            generator = torch.Generator().manual_seed(20261026 + rows)
            raw = torch.randn(rows, 64, generator=generator)
            roles = torch.arange(rows, dtype=torch.int64) % 5 if correction.routing else None
            keys = correction.keys(raw, roles)
            feedback = correction.feedback(
                ids=list(range(1, rows + 1)),
                decision_at=[5] * rows,
                available_at=[10] * rows,
                keys=keys,
                values=(torch.randn(rows, generator=generator) * 0.01).tolist(),
            )
            memory = correction.empty().write(feedback, cutoff=10)
            later = replace(
                feedback,
                ids=feedback.ids + rows,
                decision_at=feedback.decision_at + 10,
                available_at=feedback.available_at + 10,
            )

            def cycle(correction=correction, raw=raw, roles=roles, memory=memory, later=later):
                current = correction.keys(raw, roles)
                memory.read(current)
                return memory.write(replace(later, keys=current), cutoff=20)

            restored = correction.restore(memory.export())
            if not torch.equal(memory.read(keys), restored.read(keys)):
                raise AssertionError("La medida no conserva la recuperación exacta")
            cases[f"{name}/{rows}"] = dict(
                keys=timed(
                    lambda correction=correction, raw=raw, roles=roles: correction.keys(raw, roles),
                    args.warmup,
                    args.repeats,
                ),
                cycle=timed(cycle, args.warmup, args.repeats),
                numeric_state_bytes=memory.numeric_state_bytes
                if hasattr(memory, "numeric_state_bytes")
                else memory.matrix.nbytes,
                coefficient_elements=memory.matrix.numel(),
                recovery_bitwise=True,
            )
            for operation in ("keys", "cycle"):
                measure = cases[f"{name}/{rows}"][operation]
                measure["rows_per_second_at_median"] = rows / (measure["p50_ms"] / 1000)
    atomic_json(
        args.output,
        dict(
            schema_version=1,
            kind="role_binding_technical_cost",
            measured_at=datetime.now(UTC).isoformat(),
            hardware=dict(
                cpu=cpu_name(), threads=2, interop_threads=1, device="cpu", dtype="float64"
            ),
            versions=dict(
                torch=str(torch.__version__), numpy=np.__version__, python=platform.python_version()
            ),
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sources_sha256=_code(),
            timing_helper_sha256=hashlib.sha256(
                Path("benchmarks/area_memory_cost.py").read_bytes()
            ).hexdigest(),
            cases=cases,
            elapsed_s=time.perf_counter() - start,
            load_start=load,
            load_end=os.getloadavg(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            limits=[
                "Las rutas del fixture varían entre filas. No se calcula el régimen real.",
                "El pico RSS abarca todo el proceso, no una diferencia entre variantes.",
                "Los percentiles con cuarenta repeticiones no establecen un objetivo de latencia.",
                "No se han medido energía, VRAM ni transferencias. El padre CUDA no se ejecuta.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
