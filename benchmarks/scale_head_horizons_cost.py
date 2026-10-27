"""Medir en CPU el coste de F1 y F5 de #749 con formas reales, sin optimizador.

Las entradas son sintéticas con las formas de la edición v3.1: ventana de 64 sesiones con
cinco canales y presencia, 384 de noticias, 512 de gráficos, 78 de fundamentales y 420 de
macro, con ausencias en las modalidades opcionales. Cada instante tiene B activos.

Para cada modelo se recorren T instantes con grafo, se calcula su pérdida frente a etiquetas
sintéticas y se ejecuta un backward. En F5 se añade el término auxiliar de una actualización
estable, con T·B decisiones y todos sus horizontes, mediante el mismo método del entrenador. Se
mide también el registro de etiquetas de F5 por evento y los bytes que guarda. Las
repeticiones se alternan entre modelos con orden rotatorio. Al final se comprueba que ningún
parámetro ha cambiado. No hay optimizador ni pasos.
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
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from mars_titan.data.input_policy import HISTORICAL_MASKED, policy_identity
from mars_titan.data.price_windows import price_window_contract
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.titans.auxiliary_horizons import AuxiliaryHorizons
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import (
    DecisionBatch,
    FinancialInputSpec,
    validated_cpu_batch,
)
from mars_titan.models.titans.laplace_scale import LaplaceScaleHead, laplace_beta_nll
from mars_titan.training.financial_run import ChronologicalTrainer
from mars_titan.training.horizon_labels import HorizonLabels
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

START = 1_609_459_200_000_000
DAY = 86_400_000_000
WIDTHS = dict(news=384, charts=512, concepts=26, indicators=140)
# Predictor de la receta histórica con máscaras y cabeza escalar.
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
HORIZONS = AuxiliaryHorizons()
MODELS = {
    "reference_mae": {},
    "laplace_scale_head_v1": dict(laplace_scale_head=asdict(LaplaceScaleHead())),
    "auxiliary_horizons_weight_zero": dict(
        auxiliary_horizons=asdict(AuxiliaryHorizons(weight=0.0))
    ),
    "auxiliary_horizons_v1": dict(auxiliary_horizons=asdict(HORIZONS)),
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


def cpu_batches(spec, rows, instants, seed=7):
    """Bloques CPU validados de un instante con B activos, ausencias y huecos de precios."""
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
        result.append(validated_cpu_batch(raw, spec))
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


def ready_rows(cpu, instants, horizons, seed=13):
    """Decisiones de una actualización estable: T·B, cada una con todos sus horizontes."""
    generator = np.random.default_rng(seed)
    rows = []
    for block in cpu[:instants]:
        for row in range(len(block.flow_ids)):
            values = {name: np.array(block.inputs[name][row]) for name in block.inputs}
            values["presence"] = np.array(block.presence[row])
            targets = generator.normal(0, 0.03, size=len(horizons)).tolist()
            rows.append((values, list(range(len(horizons))), targets))
    return rows


def graph_pass(model, batches, targets, auxiliary):
    started = time.perf_counter()
    state = model.initial_state(batches[0].flow_ids, differentiable=True)
    prepared = []
    for batch in batches:
        result = model.prepare(batch, state, differentiable=True)
        state = result.next_state
        prepared.append(result)
    point = torch.cat([value.point_predictions for value in prepared])
    target = torch.cat(targets)
    if model.scale_head is None:
        loss = torch.nn.functional.l1_loss(point, target)
    else:
        scale = torch.cat([value.scales for value in prepared])
        loss = laplace_beta_nll(torch.stack((point, scale), dim=-1), target)
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    backward = time.perf_counter() - started
    started = time.perf_counter()
    if auxiliary is not None:
        auxiliary()
    extra = time.perf_counter() - started
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("La medida encontró un gradiente no finito")
    for value in model.parameters():
        value.grad = None
    return forward, backward, extra


def auxiliary_term(model, ready):
    """El método `_auxiliary` del entrenador sobre un registro con `ready` etiquetas."""
    if model.auxiliary_heads is None:
        return None
    config = model.config.auxiliary_horizons
    trainer = SimpleNamespace(
        predictor=model,
        horizons=config,
        device="cpu",
        recipe=SimpleNamespace(accumulation_rows=None),
    )

    def run():
        ledger = HorizonLabels(config)
        ledger.ready = list(ready)
        counters = dict(auxiliary_labels_in_loss=0, auxiliary_loss_sum=0.0)
        ChronologicalTrainer._auxiliary(
            trainer, SimpleNamespace(horizons=ledger, counters=counters)
        )
        if counters["auxiliary_labels_in_loss"] != sum(len(row[1]) for row in ready):
            raise RuntimeError("El término auxiliar no consumió todas sus etiquetas")

    return run


def ledger_cost(cpu, horizons, instants):
    """Tiempo del registro por evento en régimen estable y bytes guardados por flujo."""
    ledger, times = HorizonLabels(horizons), []
    blocks = [cpu[index % len(cpu)] for index in range(instants)]
    for index, block in enumerate(blocks):
        at = START + index * DAY
        shifted = SimpleNamespace(
            flow_ids=block.flow_ids,
            prediction_at=(at,) * len(block.flow_ids),
            inputs=block.inputs,
            presence=block.presence,
        )
        started = time.perf_counter()
        if index:
            for flow in block.flow_ids:
                ledger.mature(flow, at - DAY, 0.01)
        ledger.observe(["US"])
        ledger.remember(shifted)
        ledger.take()
        times.append(time.perf_counter() - started)
    # Las decisiones que ya entregaron sus etiquetas solo conservan la etiqueta principal.
    stored = [entry[2] for entries in ledger.flows.values() for entry in entries.values()]
    stored_bytes = sum(value.nbytes for row in stored if row is not None for value in row.values())
    # La primera exportación incluye la reserva inicial de memoria del proceso.
    exported = []
    for _ in range(3):
        started = time.perf_counter()
        payload = ledger.export()
        exported.append(time.perf_counter() - started)
    return dict(
        seconds_per_event=_summary(times[horizons.span + 1 :]),
        retained_decisions=len(stored),
        stored_decisions=sum(row is not None for row in stored),
        stored_input_bytes=stored_bytes,
        export_seconds=_summary(exported),
        export_tensor_bytes=sum(value.nbytes for value in payload["inputs"].values()),
    )


def measure(output, *, rows, instants, repetitions, warmup, threads):
    limits = ((rows, 1, 1024), (instants, 1, 16), (repetitions, 1, 20), (warmup, 0, 5))
    if any(type(value) is not int or not low <= value <= high for value, low, high in limits):
        raise ValueError("Las formas o repeticiones exceden el presupuesto de la medida")
    if Path(output).exists():
        raise FileExistsError("La medida necesita un recibo nuevo")
    started = time.perf_counter()
    load_at_start = os.getloadavg()
    torch.set_num_threads(threads)
    torch.backends.mha.set_fastpath_enabled(False)
    policy = apply_kernel_policy(FP32_STRICT)
    spec = specification()
    cpu = cpu_batches(spec, rows, instants)
    inputs = [DecisionBatch.from_validated(block, dtype=torch.float32) for block in cpu]
    generator = torch.Generator().manual_seed(11)
    targets = [torch.randn(rows, generator=generator) * 0.015 for _ in range(instants)]
    ready = ready_rows(cpu, instants, HORIZONS.horizons)
    ledger = ledger_cost(cpu, HORIZONS, 3 * HORIZONS.span)
    built = {}
    for name, changes in MODELS.items():
        config = FinancialConfig(spec, variant="mac_online", seed=42, **(BASE | changes))
        model = FinancialPredictor(config, dtype=torch.float32).train()
        built[name] = (model, _digest(model), auxiliary_term(model, ready))
    samples = {name: [] for name in built}
    with sdpa_kernel(SDPBackend.MATH):
        for _ in range(warmup):
            for model, _, auxiliary in built.values():
                graph_pass(model, inputs, targets, auxiliary)
        names = list(built)
        for repetition in range(repetitions):
            shift = repetition % len(names)
            for name in names[shift:] + names[:shift]:
                model, _, auxiliary = built[name]
                samples[name].append(graph_pass(model, inputs, targets, auxiliary))
    models = {}
    for name, (model, before, auxiliary) in built.items():
        if _digest(model) != before:
            raise RuntimeError("La medida ha modificado parámetros")
        totals = [sum(value) for value in samples[name]]
        models[name] = dict(
            predictor_changes=MODELS[name],
            total_parameters=sum(value.numel() for value in model.parameters()),
            forward_with_graph_seconds=_summary([value[0] for value in samples[name]]),
            backward_seconds=_summary([value[1] for value in samples[name]]),
            auxiliary_term_seconds=_summary([value[2] for value in samples[name]]),
            auxiliary_decisions_per_update=len(ready) if auxiliary is not None else 0,
            auxiliary_labels_per_update=sum(len(row[1]) for row in ready) if auxiliary else 0,
            update_seconds=_summary(totals),
            parameters_unchanged=True,
        )
        print(name, round(statistics.median(totals), 4), flush=True)
    control = models["reference_mae"]
    for value in models.values():
        value["added_parameters"] = value["total_parameters"] - control["total_parameters"]
        value["update_relative_to_control"] = (
            value["update_seconds"]["median"] / control["update_seconds"]["median"]
        )
    receipt = dict(
        schema_version=1,
        kind="scale_head_horizons_cost",
        issue=749,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Forward con grafo de T instantes, pérdida y backward de cada modelo, más el término "
            "auxiliar de F5 de una actualización estable. Entradas sintéticas con las formas de "
            "la edición v3.1. Sin optimizador ni pasos."
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
                Path("src/mars_titan/models/titans/laplace_scale.py"),
                Path("src/mars_titan/models/titans/auxiliary_horizons.py"),
                Path("src/mars_titan/models/titans/financial.py"),
                Path("src/mars_titan/training/financial_run.py"),
                Path("src/mars_titan/training/horizon_labels.py"),
            )
        },
        threads=torch.get_num_threads(),
        shapes=dict(rows=rows, instants=instants, hidden_size=BASE["hidden_size"], **WIDTHS),
        base_predictor=BASE,
        horizons=HORIZONS.identity(),
        repetitions=repetitions,
        warmup=warmup,
        order="repeticiones alternadas entre modelos con orden rotatorio",
        models=models,
        ledger=ledger,
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
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
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
    )


if __name__ == "__main__":
    main()
