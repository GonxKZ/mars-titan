"""Medir en CPU el coste de R1, U1 y U2 de #786 frente al núcleo, sin optimizador.

Reutiliza las entradas sintéticas de `objective_variants_cost.py`, con las formas de la edición
v3.1 y B activos de un mercado por instante. Cada modelo de las tres declaraciones recorre T
instantes con grafo, calcula el error absoluto frente a objetivos sintéticos y ejecuta un
backward. También se mide la inferencia sin grafo. Las repeticiones alternan los modelos con
orden rotatorio y al final se comprueba que ningún parámetro ha cambiado. Aparte se registra,
en una pasada sin grafo, la fracción de canales que quedan fuera del umbral de Huber al
empezar con estas entradas. No hay optimizador.
"""

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.objective_variants_cost import (
    BASE,
    _commit,
    _digest,
    _summary,
    batches,
    specification,
)
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.titans import neural_memory
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

ROOT = Path(__file__).resolve().parents[1]
DECLARATIONS = (
    "configs/experiments/huber-memory-v1.json",
    "configs/experiments/differential-attention-v1.json",
    "configs/experiments/scale-invariant-time-v1.json",
)


def arms():
    """La referencia común y la variante de cada declaración, en ese orden."""
    result = {}
    for path in DECLARATIONS:
        declaration = json.loads((ROOT / path).read_text(encoding="utf-8"))
        for name, arm in declaration["arms"].items():
            if result.setdefault(name, arm["predictor_changes"]) != arm["predictor_changes"]:
                raise ValueError("Dos declaraciones dan cambios distintos al mismo modelo")
    return result


def walk(model, inputs, *, differentiable):
    state = model.initial_state(inputs[0].flow_ids, differentiable=differentiable)
    prepared = []
    for batch in inputs:
        result = model.prepare(batch, state, differentiable=differentiable, session_complete=True)
        state = result.next_state
        prepared.append(result)
    return prepared


def graph_pass(model, inputs, targets):
    started = time.perf_counter()
    prepared = walk(model, inputs, differentiable=True)
    point = torch.cat([value.point_predictions for value in prepared])
    loss = torch.nn.functional.l1_loss(point, torch.cat(targets))
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("La medida encontró un gradiente no finito")
    backward = time.perf_counter() - started
    for value in model.parameters():
        value.grad = None
    return forward, backward


def inference(model, inputs):
    started = time.perf_counter()
    with torch.no_grad():
        walk(model, inputs, differentiable=False)
    return time.perf_counter() - started


def clipped_fraction(model, inputs):
    """Fracción de canales con |r| > δ en la escritura de cada token, sin grafo."""
    original, counts = neural_memory.huber_elements, [0, 0]

    def counted(residual, threshold):
        counts[0] += int((residual.detach().abs() > threshold.detach()).sum())
        counts[1] += residual.numel()
        return original(residual, threshold)

    neural_memory.huber_elements = counted
    try:
        inference(model, inputs)
    finally:
        neural_memory.huber_elements = original
    return counts[0] / counts[1]


def measure(output, *, rows, instants, repetitions, warmup, threads):
    limits = (
        (rows, 1, 256),
        (instants, 2, 16),
        (repetitions, 1, 20),
        (warmup, 0, 5),
        (threads, 1, 8),
    )
    if any(type(value) is not int or not low <= value <= high for value, low, high in limits):
        raise ValueError("Las formas, repeticiones o hilos exceden el presupuesto de la medida")
    if Path(output).exists():
        raise FileExistsError("La medida necesita un recibo nuevo")
    started = time.perf_counter()
    load_at_start = os.getloadavg()
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(False)
    policy = apply_kernel_policy(FP32_STRICT)
    with sdpa_kernel(SDPBackend.MATH):
        return _measure(
            output,
            rows=rows,
            instants=instants,
            repetitions=repetitions,
            warmup=warmup,
            threads=threads,
            policy=policy,
            started=started,
            load_at_start=load_at_start,
        )


def _measure(
    output, *, rows, instants, repetitions, warmup, threads, policy, started, load_at_start
):
    spec = specification()
    inputs = batches(spec, rows, instants)
    generator = torch.Generator().manual_seed(11)
    targets = [torch.randn(rows, generator=generator) * 0.015 for _ in range(instants)]
    declared = arms()
    names = list(declared)
    built = {}
    for name, changes in declared.items():
        config = FinancialConfig(spec, variant="mac_online", seed=42, **(BASE | changes))
        model = FinancialPredictor(config, dtype=torch.float32).train()
        built[name] = (model, _digest(model))
    for _ in range(warmup):
        for model, _ in built.values():
            graph_pass(model, inputs, targets)
            inference(model, inputs)
    graph = {name: [] for name in built}
    plain = {name: [] for name in built}
    for repetition in range(repetitions):
        shift = repetition % len(names)
        for name in names[shift:] + names[:shift]:
            model = built[name][0]
            graph[name].append(graph_pass(model, inputs, targets))
            plain[name].append(inference(model, inputs))
    models = {}
    for name, (model, before) in built.items():
        if _digest(model) != before:
            raise RuntimeError("La medida ha modificado parámetros")
        models[name] = dict(
            predictor_changes=declared[name],
            total_parameters=sum(value.numel() for value in model.parameters()),
            forward_with_graph_seconds=_summary([value[0] for value in graph[name]]),
            backward_seconds=_summary([value[1] for value in graph[name]]),
            inference_seconds=_summary(plain[name]),
            parameters_unchanged=True,
        )
    control = models[names[0]]
    for value in models.values():
        value["added_parameters"] = value["total_parameters"] - control["total_parameters"]
        value["relative_to_control"] = {
            key: value[key]["median"] / control[key]["median"]
            for key in ("forward_with_graph_seconds", "backward_seconds", "inference_seconds")
        }
    huber = [name for name, changes in declared.items() if "memory_huber" in changes]
    receipt = dict(
        schema_version=1,
        kind="core_variants_cost",
        issue=786,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Forward con grafo de T instantes, error absoluto y backward, más inferencia sin "
            "grafo, con entradas sintéticas con las formas de la edición v3.1. Sin optimizador "
            "ni pasos."
        ),
        device="cpu",
        cpu=cpu_name(),
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        precision="float32",
        kernel_policy=policy,
        threads=threads,
        shape=dict(rows=rows, instants=instants, hidden_size=BASE["hidden_size"]),
        repetitions=repetitions,
        warmup=warmup,
        load_average=dict(start=list(load_at_start), end=list(os.getloadavg())),
        models=models,
        huber_clipped_fraction_at_start={
            name: clipped_fraction(built[name][0], inputs) for name in huber
        },
        elapsed_seconds=time.perf_counter() - started,
    )
    atomic_json(Path(output), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--instants", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=2)
    arguments = parser.parse_args()
    receipt = measure(
        arguments.output,
        rows=arguments.rows,
        instants=arguments.instants,
        repetitions=arguments.repetitions,
        warmup=arguments.warmup,
        threads=arguments.threads,
    )
    for name, model in receipt["models"].items():
        print(name, json.dumps(model["relative_to_control"]))
    print("huber", json.dumps(receipt["huber_clipped_fraction_at_start"]))


if __name__ == "__main__":
    main()
