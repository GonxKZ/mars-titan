"""Medir en CPU el coste de la fiabilidad madura de #791 y de sus controles, sin optimizador.

Reutiliza las entradas sintéticas de `objective_variants_cost.py`, con las formas de la edición
v3.1 y B activos de un mercado por instante. Cada modelo de la declaración recorre T instantes
con grafo. Antes de cada instante maduran las etiquetas sintéticas del anterior y actualizan la
señal, que entra en `prepare` como en el recorrido. El control de sorpresa registra además la
pérdida asociativa de las filas emitidas. Después se calcula el error absoluto frente a los
objetivos sintéticos y se ejecuta un backward. También se mide la inferencia sin grafo y, por
separado, el acumulador por evento con 64, 256 y 1.024 etiquetas. Las repeticiones alternan los
modelos con orden rotatorio y al final se comprueba que ningún parámetro ha cambiado. No hay
optimizador.
"""

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
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
from mars_titan.memory.mature_reliability import ReliabilitySignal
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.reliability_gates import MatureReliability
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs/experiments/mature-reliability-signal-v1.json"
EVENT_ROWS = (64, 256, 1024)


def arms():
    """Modelos de la declaración en su orden, empezando por la referencia."""
    declaration = json.loads(DECLARATION.read_text(encoding="utf-8"))
    return {name: arm["predictor_changes"] for name, arm in declaration["arms"].items()}


def walk(model, inputs, targets, *, differentiable):
    """Recorrer los instantes con la señal alimentada como en el recorrido cronológico."""
    config = model.config.mature_reliability
    signal = None if config is None else ReliabilitySignal(config)
    state = model.initial_state(inputs[0].flow_ids, differentiable=differentiable)
    prepared = []
    for index, batch in enumerate(inputs):
        options = {}
        if signal is not None:
            if prepared:
                # Las etiquetas de la decisión anterior maduran antes de emitir esta.
                issued = prepared[-1].point_predictions.detach().tolist()
                outcomes = targets[index - 1].tolist()
                signal.settle(zip(batch.flow_ids, issued, outcomes, strict=True))
            weight = model.head.weight
            options["reliability"] = torch.tensor(
                signal.values(batch.flow_ids), dtype=weight.dtype, device=weight.device
            )
        result = model.prepare(
            batch, state, differentiable=differentiable, session_complete=True, **options
        )
        if result.surprise is not None:
            signal.observe(batch.flow_ids, result.surprise.tolist())
        state = result.next_state
        prepared.append(result)
    return prepared


def graph_pass(model, inputs, targets):
    started = time.perf_counter()
    prepared = walk(model, inputs, targets, differentiable=True)
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


def inference(model, inputs, targets):
    started = time.perf_counter()
    with torch.no_grad():
        walk(model, inputs, targets, differentiable=False)
    return time.perf_counter() - started


def accumulator_cost(repetitions):
    """Microsegundos por evento de la señal sola: aplicar etiquetas o pérdidas y leerla."""
    result = {}
    for name, changes in arms().items():
        if not changes:
            continue
        config = MatureReliability(**changes["mature_reliability"])
        for count in EVENT_ROWS:
            flows = [f"US/S{asset:04}" for asset in range(count)]
            generator = np.random.default_rng(count)
            signal, update, read = ReliabilitySignal(config), [], []
            for _ in range(repetitions):
                first, second = generator.normal(size=(2, count)).tolist()
                started = time.perf_counter_ns()
                if config.source == "titans_surprise":
                    signal.observe(flows, [abs(value) for value in first])
                else:
                    signal.settle(zip(flows, first, second, strict=True))
                update.append((time.perf_counter_ns() - started) / 1e3)
                started = time.perf_counter_ns()
                signal.values(flows)
                read.append((time.perf_counter_ns() - started) / 1e3)
            result[f"{name}/{count}"] = dict(
                update_microseconds=_summary(update), read_microseconds=_summary(read)
            )
    return result


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
            inference(model, inputs, targets)
    graph = {name: [] for name in built}
    plain = {name: [] for name in built}
    for repetition in range(repetitions):
        shift = repetition % len(names)
        for name in names[shift:] + names[:shift]:
            model = built[name][0]
            graph[name].append(graph_pass(model, inputs, targets))
            plain[name].append(inference(model, inputs, targets))
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
    receipt = dict(
        schema_version=1,
        kind="mature_reliability_cost",
        issue=791,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Forward con grafo de T instantes con la señal alimentada por etiquetas sintéticas "
            "que maduran en el instante siguiente, error absoluto y backward, más inferencia sin "
            "grafo y el acumulador por evento. Entradas sintéticas con las formas de la edición "
            "v3.1. Sin optimizador ni pasos."
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
        accumulator=accumulator_cost(repetitions),
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


if __name__ == "__main__":
    main()
