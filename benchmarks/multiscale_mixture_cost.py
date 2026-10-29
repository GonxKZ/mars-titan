"""Coste técnico de E1 sobre claves y etiquetas de un fixture, sin predictor ni optimizador.

La primera cohorte publica en t=0. Los componentes pendientes se calculan en el día 7.
Cada ciclo medido consulta en el día 14 y después publica la etiqueta de aquella decisión,
usando los componentes guardados del día 7. No mide error financiero ni calcula el régimen.
Con la declaración de R2 (#786) el día 14 cae en otra semana ISO y en el mismo mes, así que
cada ciclo cierra el periodo semanal con su paso proximal y amplía el mensual.
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

from benchmarks.area_memory_cost import state_bytes, timed
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.mars_titan_variant import check_components, load_declaration
from mars_titan.memory.multiscale_mixture import DAY_US
from mars_titan.training.mars_titan_correction import (
    _code,
    arm_components,
    case_values,
    load_correction_recipe,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument(
        "--declaration",
        type=Path,
        default=Path("configs/experiments/multiscale-mature-mixture-v1.json"),
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("El recibo debe ser nuevo")
    if not 0 <= args.warmup <= 10 or not 1 <= args.repeats <= 100:
        raise ValueError("El calentamiento o las repeticiones exceden sus límites")
    if torch.cuda.is_initialized():
        raise RuntimeError("Esta medida de correctores CPU no usa CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    path = args.declaration
    study = json.loads(path.read_text())
    declaration = load_declaration()
    start, load = time.perf_counter(), os.getloadavg()
    cases = {}
    recipes = {}
    for name, arm in (study["arms"] | study.get("references", {})).items():
        recipe = load_correction_recipe(arm["recipe"])
        recipes[arm["recipe"]] = hashlib.sha256(Path(arm["recipe"]).read_bytes()).hexdigest()
        _, correction = check_components(
            declaration, arm_components(arm["components"], case_values(recipe, "eta5e-2"))
        )
        for rows in (64, 256, 1024):
            generator = torch.Generator().manual_seed(20261026 + rows)
            raw = torch.randn(rows, 64, generator=generator)
            routes = (
                torch.arange(rows, dtype=torch.int64) % 5
                if correction.routing is not None
                else None
            )
            keys = correction.keys(raw, routes)
            memory = correction.empty()
            feedback = correction.feedback(
                ids=list(range(1, rows + 1)),
                decision_at=[-1] * rows,
                available_at=[0] * rows,
                keys=keys,
                values=(torch.randn(rows, generator=generator) * 0.01).tolist(),
            )
            if correction.multiscale is not None:
                _, initial = memory.predict(keys, at=0)
                memory = memory.write(feedback, cutoff=0, issued_predictions=initial)
                _, issued = memory.predict(keys, at=7 * DAY_US)
            else:
                memory = memory.write(feedback, cutoff=0)
                issued = None
            later = replace(
                feedback,
                ids=feedback.ids + rows,
                decision_at=torch.full((rows,), 7 * DAY_US, dtype=torch.int64),
                available_at=torch.full((rows,), 14 * DAY_US, dtype=torch.int64),
            )

            def cycle(
                memory=memory,
                correction=correction,
                raw=raw,
                routes=routes,
                later=later,
                issued=issued,
            ):
                current = correction.keys(raw, routes)
                if issued is not None:
                    memory.predict(current, at=14 * DAY_US)
                    return memory.write(
                        replace(later, keys=current), cutoff=14 * DAY_US, issued_predictions=issued
                    )
                memory.read(current)
                return memory.write(replace(later, keys=current), cutoff=14 * DAY_US)

            restored = correction.restore(memory.export())
            expected = (
                memory.read(keys, at=14 * DAY_US) if issued is not None else memory.read(keys)
            )
            actual = (
                restored.read(keys, at=14 * DAY_US) if issued is not None else restored.read(keys)
            )
            if not torch.equal(actual, expected):
                raise AssertionError("La recuperación no coincide bit a bit")
            measured = timed(cycle, args.warmup, args.repeats)
            measured["rows_per_second_at_median"] = rows / (measured["p50_ms"] / 1000)
            cases[name + "/" + str(rows)] = dict(
                cycle=measured,
                experts=0 if issued is None else issued.shape[1],
                numeric_state_bytes=memory.numeric_state_bytes
                if hasattr(memory, "numeric_state_bytes")
                else state_bytes(memory.export()),
                serialized_tensor_bytes=state_bytes(memory.export()),
                pending_component_bytes_per_row=0 if issued is None else issued.shape[1] * 8,
                open_period_rows=memory.pending_rows if issued is not None else None,
                recovery_bitwise=True,
            )
    atomic_json(
        args.output,
        dict(
            schema_version=1,
            kind="multiscale_mixture_technical_cost",
            measured_at=datetime.now(UTC).isoformat(),
            hardware=dict(
                cpu=cpu_name(), device="cpu", dtype="float64", threads=2, interop_threads=1
            ),
            versions=dict(
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__
            ),
            declaration=str(path),
            configuration_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            sources_sha256=_code(),
            recipes_sha256=recipes,
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            timing_helper_sha256=hashlib.sha256(
                Path("benchmarks/area_memory_cost.py").read_bytes()
            ).hexdigest(),
            cases=cases,
            elapsed_s=time.perf_counter() - start,
            load_start=load,
            load_end=os.getloadavg(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            limits=[
                "Fixture con rutas ya dadas, sin predictor, cálculo del régimen o almacenamiento.",
                "Los bytes numéricos excluyen colas, cabeceras y temporales.",
                "Los componentes pendientes se informan por fila.",
                "El pico RSS abarca el proceso completo, no una diferencia entre configuraciones.",
                "Las cuarenta muestras no establecen un objetivo de latencia de cola.",
                "No se mide energía o VRAM.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
