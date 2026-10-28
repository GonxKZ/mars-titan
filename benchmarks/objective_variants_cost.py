"""Medir en CPU el coste de las variantes de objetivo y cabezas de #749, sin optimizador.

Las entradas son sintéticas con las formas reales de la edición v3.1: ventana de 64 sesiones
con cinco canales y presencia, 384 de noticias, 512 de gráficos, 78 de fundamentales y 420 de
macro, con ausencias en todas las modalidades opcionales. Cada instante tiene B activos de un
mismo mercado, que forman una sesión completa en un solo bloque, como en el recorrido.

Para cada modelo se recorren T instantes con grafo, se calcula su pérdida frente a etiquetas
sintéticas y se ejecuta un backward. Se mide también la inferencia sin grafo y el pico neto
del asignador de ATen en CPU con el perfilador. Las repeticiones se alternan entre modelos con
orden rotatorio. Al final se comprueba que ningún parámetro ha cambiado. No hay optimizador.
"""

import argparse
import hashlib
import os
import platform
import resource
import statistics
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.price_windows import price_window_contract
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch, FinancialInputSpec
from mars_titan.models.titans.session_ic_objective import SessionICObjective
from mars_titan.models.titans.session_ranks import SESSION_RANK_INPUTS
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

START = 1_609_459_200_000_000
DAY = 86_400_000_000
WIDTHS = dict(news=384, charts=512, concepts=26, indicators=140)
# Predictor de la receta histórica con máscaras, con la cabeza escalar que sustituyen los
# objetivos de #749.
BASE = dict(
    hidden_size=64,
    layers=1,
    persistent_tokens=4,
    max_batch=1024,
    max_state_bytes=134217728,
    gate_bias=dict(alpha_half_life=256.0, eta=0.5, theta=0.05),
    memory_residual_layer_norm=True,
    attention_layout="sequence_contiguous_v1",
)
MODELS = {
    "reference_mae": {},
    SESSION_RANK_INPUTS: dict(session_rank_inputs=SESSION_RANK_INPUTS),
    "session_ic_objective_v1": dict(session_ic_objective=asdict(SessionICObjective())),
    "session_rank_inputs_and_ic_objective": dict(
        session_rank_inputs=SESSION_RANK_INPUTS,
        session_ic_objective=asdict(SessionICObjective()),
    ),
}


def _commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def specification():
    representation = dict(
        **policy_identity(HISTORICAL_MASKED),
        fundamental_concepts=[f"synthetic:C{i}" for i in range(WIDTHS["concepts"])],
        macro_indicators=[f"macro_{i}" for i in range(WIDTHS["indicators"])],
        encoders={"purpose": "synthetic_cost"},
        representation_code={"synthetic": "0" * 64},
        text_aggregation="mean",
        context_sessions=64,
        news_lookback_sessions=5,
        price_window=price_window_contract(
            {"US": dict(start="2000-01-03", end="2023-12-29", decisions_sha256="0" * 64)},
            {"US": []},
        ),
    )
    dimensions = dict(
        prices=6,
        news=WIDTHS["news"],
        charts=WIDTHS["charts"],
        fundamentals=3 * WIDTHS["concepts"],
        macro=3 * WIDTHS["indicators"],
    )
    return FinancialInputSpec(
        source_sha256="a" * 64,
        view_sha256="b" * 64,
        representation=representation,
        dimensions=dimensions,
        input_policy=HISTORICAL_MASKED,
    )


def _numeric(generator, rows, width, observed, fraction):
    masks = (generator.random((rows, width)) < fraction) & observed[:, None]
    masks[observed, 0] = True
    values = np.where(masks, generator.normal(size=(rows, width)), 0)
    ages = np.where(masks, generator.random((rows, width)), 0)
    return np.concatenate([values, masks, ages], axis=1).astype(np.float32)


def batches(spec, rows, instants, seed=7, *, device="cpu"):
    """Lotes de un instante con B activos, ausencias y huecos de la ventana de precios."""
    generator = np.random.default_rng(seed)
    result = []
    for index in range(instants):
        at = START + index * DAY
        presence = np.ones((rows, 5), dtype=np.bool_)
        presence[:, 1] = generator.random(rows) < 0.4
        presence[:, 3] = generator.random(rows) < 0.7
        presence[:, 4] = generator.random(rows) < 0.9
        prices = generator.normal(0, 0.02, size=(rows, 64, 6)).astype(np.float32)
        prices[..., 5] = 1
        gaps = generator.random((rows, 64)) < 0.01
        gaps[:, -1] = False
        prices[gaps] = 0
        news = generator.normal(size=(rows, WIDTHS["news"])).astype(np.float32)
        news[~presence[:, 1]] = 0
        raw = dict(
            inputs=dict(
                prices=prices,
                news=news,
                charts=generator.normal(size=(rows, WIDTHS["charts"])).astype(np.float32),
                fundamentals=_numeric(generator, rows, WIDTHS["concepts"], presence[:, 3], 0.6),
                macro=_numeric(generator, rows, WIDTHS["indicators"], presence[:, 4], 0.8),
            ),
            presence=presence,
            sample_ids=[f"US/S{asset:04}/{at}" for asset in range(rows)],
            market=["US"] * rows,
            prediction_at=np.full(rows, at, dtype="datetime64[us]"),
            input_available_at=np.full(rows, at - 1, dtype="datetime64[us]"),
        )
        result.append(DecisionBatch.from_corpus(raw, spec, dtype=torch.float32, device=device))
    return result


def _digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _summary(values):
    ordered = sorted(values)
    return dict(
        median=statistics.median(ordered),
        minimum=ordered[0],
        maximum=ordered[-1],
        samples=list(values),
    )


def _loss(model, prepared, targets, batches):
    rows = [(batch.flow_ids, batch.prediction_at) for batch in batches]
    target = torch.cat(targets)
    if model.output_objective is None:
        point = torch.cat([value.point_predictions for value in prepared])
        return torch.nn.functional.l1_loss(point, target)
    outputs = torch.cat([value.outputs for value in prepared])
    sessions = [("US", at) for flows, moments in rows for at in moments]
    return model.output_objective.loss(outputs, target, sessions)


def graph_pass(model, batches, targets):
    started = time.perf_counter()
    state = model.initial_state(batches[0].flow_ids, differentiable=True)
    prepared = []
    for batch in batches:
        result = model.prepare(batch, state, differentiable=True, session_complete=True)
        state = result.next_state
        prepared.append(result)
    loss = _loss(model, prepared, targets, batches)
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("La medida encontró un gradiente no finito")
    backward = time.perf_counter() - started
    for value in model.parameters():
        value.grad = None
    return forward, backward


def inference(model, batches):
    started = time.perf_counter()
    with torch.no_grad():
        state = model.initial_state(batches[0].flow_ids)
        for batch in batches:
            state = model.prepare(batch, state, session_complete=True).next_state
    return time.perf_counter() - started


def allocated_peak(model, batches, targets):
    """Pico neto del asignador de ATen en CPU durante un recorrido con grafo y su backward."""
    activities = [torch.profiler.ProfilerActivity.CPU]
    with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
        graph_pass(model, batches, targets)
    events = sorted(
        (event.start_ns(), event.nbytes())
        for event in profile.profiler.kineto_results.events()
        if event.name() == "[memory]" and event.device_type() == torch.autograd.DeviceType.CPU
    )
    current = peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def measure(output, *, rows, instants, repetitions, warmup, threads, names):
    limits = (
        (rows, 1, 256),
        (instants, 1, 16),
        (repetitions, 1, 20),
        (warmup, 0, 5),
        (threads, 1, 8),
    )
    if any(type(value) is not int or not low <= value <= high for value, low, high in limits):
        raise ValueError("Las formas, repeticiones o hilos exceden el presupuesto de la medida")
    if (
        not names
        or names[0] != "reference_mae"
        or len(names) != len(set(names))
        or not set(names) <= set(MODELS)
    ):
        raise ValueError("La medida necesita primero el control y modelos conocidos sin duplicar")
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
            names=names,
            policy=policy,
            started=started,
            load_at_start=load_at_start,
        )


def _measure(
    output, *, rows, instants, repetitions, warmup, threads, names, policy, started, load_at_start
):
    spec = specification()
    inputs = batches(spec, rows, instants)
    generator = torch.Generator().manual_seed(11)
    targets = [torch.randn(rows, generator=generator) * 0.015 for _ in range(instants)]
    built = {}
    for name in names:
        config = FinancialConfig(spec, variant="mac_online", seed=42, **(BASE | MODELS[name]))
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
        peak = allocated_peak(model, inputs, targets)
        if _digest(model) != before:
            raise RuntimeError("La medida ha modificado parámetros")
        models[name] = dict(
            predictor_changes=MODELS[name],
            total_parameters=sum(value.numel() for value in model.parameters()),
            forward_with_graph_seconds=_summary([value[0] for value in graph[name]]),
            backward_seconds=_summary([value[1] for value in graph[name]]),
            inference_seconds=_summary(plain[name]),
            allocated_peak_bytes=peak,
            parameters_unchanged=True,
        )
        print(name, round(models[name]["forward_with_graph_seconds"]["median"], 4), flush=True)
    control = models[names[0]]
    for value in models.values():
        value["added_parameters"] = value["total_parameters"] - control["total_parameters"]
        value["relative_to_control"] = {
            key: value[key]["median"] / control[key]["median"]
            for key in ("forward_with_graph_seconds", "backward_seconds", "inference_seconds")
        } | dict(
            allocated_peak_bytes=value["allocated_peak_bytes"] / control["allocated_peak_bytes"]
        )
    receipt = dict(
        schema_version=1,
        kind="objective_variants_cost",
        issue=749,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Forward con grafo de T instantes, pérdida de cada objetivo frente a etiquetas "
            "sintéticas y backward, más inferencia sin grafo. Entradas sintéticas con las formas "
            "de la edición v3.1. Sin optimizador ni pasos."
        ),
        device="cpu",
        cpu=cpu_name(),
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        precision="float32",
        kernel_policy=policy,
        sdpa_backend="MATH",
        source_sha256={
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).relative_to(Path.cwd()),
                Path("src/mars_titan/models/titans/session_ranks.py"),
                Path("src/mars_titan/models/titans/session_ic_objective.py"),
                Path("src/mars_titan/models/titans/financial.py"),
            )
        },
        threads=torch.get_num_threads(),
        shapes=dict(rows=rows, instants=instants, hidden_size=BASE["hidden_size"], **WIDTHS),
        base_predictor=BASE,
        repetitions=repetitions,
        warmup=warmup,
        order="repeticiones alternadas entre modelos con orden rotatorio",
        memory="pico neto del asignador de ATen en CPU con el perfilador, una pasada con grafo",
        models=models,
        optimizer_steps=0,
        training_runs=0,
        evaluation=False,
        total_wall_seconds=time.perf_counter() - started,
        process_peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        load_average_start=load_at_start,
        load_average_end=os.getloadavg(),
    )
    atomic_json(Path(output), receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--instants", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("La medida se hace en CPU, con CUDA_VISIBLE_DEVICES=-1")
    measure(
        args.output,
        rows=args.rows,
        instants=args.instants,
        repetitions=args.repetitions,
        warmup=args.warmup,
        threads=args.threads,
        names=args.models,
    )


if __name__ == "__main__":
    main()
