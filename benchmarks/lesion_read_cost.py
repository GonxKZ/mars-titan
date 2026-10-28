"""Coste técnico de las lecturas B4, sin predictor, optimizador ni datos científicos.

Cada perfil usa un estado escrito con 64 resultados de prueba y consultas de 64, 256 y
1.024 filas. El recorrido diagnóstico calcula la lectura original y veinte condiciones
sobre ese mismo estado. Incluye el codec y las máscaras. No mide aún el registro durable.
"""

import argparse
import hashlib
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
from mars_titan.memory import (
    area_memory,
    area_query_bridge,
    associative_memory,
    lesion_protocol,
    multiscale_mixture,
    role_binding,
)
from mars_titan.memory.area_memory import AreaLayout
from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
from mars_titan.memory.lesion_protocol import LesionProtocol
from mars_titan.memory.multiscale_mixture import DAY_US, MultiscaleMixtureConfig
from mars_titan.memory.role_binding import RoleBindingConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("El recibo debe ser nuevo")
    if not 0 <= args.warmup <= 10 or not 1 <= args.repeats <= 100:
        raise ValueError("El calentamiento o las repeticiones exceden sus límites")
    if torch.cuda.is_initialized():
        raise RuntimeError("La referencia B4 usa los correctores CPU y no admite otra carga CUDA")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    base = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25), areas=AreaLayout())
    bridge = replace(base, query_bridge=AreaQueryBridgeConfig())
    role = replace(bridge, role_binding=RoleBindingConfig())
    profiles = dict(
        areas=base,
        monolithic=replace(base, areas=AreaLayout(areas=1, shifts=4)),
        bridge=bridge,
        bridge_four_steps=replace(bridge, query_bridge=AreaQueryBridgeConfig(steps=4)),
        roles=role,
        multiscale=replace(role, multiscale=MultiscaleMixtureConfig()),
    )
    started, load = time.perf_counter(), os.getloadavg()
    results, plans = {}, {}
    for name, correction in profiles.items():
        plan = LesionProtocol().bind(correction)
        plans[name] = plan.identity()
        generator = np.random.Generator(np.random.PCG64(20261026))
        raw = generator.normal(size=(64, 64)).astype(np.float32)
        routes = torch.arange(64, dtype=torch.int64) % 5 if correction.routing else None
        keys = correction.keys(raw, routes)
        feedback = correction.feedback(
            ids=list(range(1, 65)),
            decision_at=[0] * 64,
            available_at=[1] * 64,
            keys=keys,
            values=(generator.normal(size=64) * 0.01).tolist(),
        )
        state = correction.empty()
        extra = (
            dict(issued_predictions=torch.zeros(64, 3, dtype=torch.float64))
            if correction.multiscale
            else {}
        )
        state = state.write(feedback, cutoff=1, **extra)
        before = state.export()
        clock = dict(at=7 * DAY_US + 1) if correction.multiscale else {}
        for rows in (64, 256, 1024):
            raw = (
                np.random.Generator(np.random.PCG64(20261026 + rows))
                .normal(size=(rows, 64))
                .astype(np.float32)
            )
            routes = torch.arange(rows, dtype=torch.int64) % 5 if correction.routing else None

            def original(raw=raw, routes=routes, state=state, correction=correction, clock=clock):
                return state.read(correction.keys(raw, routes), **clock)

            def diagnostic(
                raw=raw, routes=routes, state=state, correction=correction, clock=clock, plan=plan
            ):
                query = correction.keys(raw, routes)
                prediction = state.read(query, **clock)
                return plan.read(state, query, at=7 * DAY_US + 1, baseline=prediction)

            actual = diagnostic()
            if not torch.equal(actual[:, :1], original()) or torch.count_nonzero(actual[:, 1]):
                raise AssertionError("Los controles intacto y cero no conservan su contrato")
            normal = timed(original, args.warmup, args.repeats)
            lesion = timed(diagnostic, args.warmup, args.repeats)
            normal["rows_per_second_at_median"] = rows / (normal["p50_ms"] / 1000)
            lesion["rows_per_second_at_median"] = rows / (lesion["p50_ms"] / 1000)
            results[f"{name}/{rows}"] = dict(
                original=normal,
                diagnostic=lesion,
                conditions=len(plan.names),
                output_bytes=actual.nbytes,
                raw_query_bytes=raw.nbytes,
                numeric_state_bytes=state.numeric_state_bytes,
            )
        if state.export()["matrix_sha256"] != before["matrix_sha256"] or (
            correction.multiscale and state.export()["state_sha256"] != before["state_sha256"]
        ):
            raise AssertionError("La medida modificó el estado original")
    modules = (
        area_memory,
        area_query_bridge,
        associative_memory,
        role_binding,
        multiscale_mixture,
        lesion_protocol,
    )
    atomic_json(
        args.output,
        dict(
            schema_version=1,
            kind="lesion_read_technical_cost",
            measured_at=datetime.now(UTC).isoformat(),
            hardware=dict(cpu=cpu_name(), device="cpu", cpu_threads=2, interop_threads=1),
            versions=dict(
                python=platform.python_version(), torch=str(torch.__version__), numpy=np.__version__
            ),
            sources_sha256={
                module.__name__: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for module in modules
            },
            benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            timing_helper_sha256=hashlib.sha256(
                Path("benchmarks/area_memory_cost.py").read_bytes()
            ).hexdigest(),
            plans=plans,
            cases=results,
            warmup=args.warmup,
            elapsed_s=time.perf_counter() - started,
            load_start=load,
            load_end=os.getloadavg(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            limits=[
                "Fixture de un estado maduro, sin entrenamiento ni métricas predictivas reales.",
                "Incluye codec, lectura original, máscaras y lecturas diagnósticas secuenciales.",
                "Excluye predictor, régimen, registro durable, colas y transferencias GPU.",
                "Los picos RSS son del proceso entero y los bytes de estado excluyen temporales.",
                "Quince muestras no permiten fijar un objetivo de latencia de cola.",
                "No se mide energía ni VRAM.",
            ],
        ),
    )


if __name__ == "__main__":
    main()
