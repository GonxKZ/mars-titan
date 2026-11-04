"""Medir en CPU el coste técnico de F2 y F3 de #749, sin optimizador ni pasos.

Usa el predictor de la receta histórica y las entradas sintéticas con las formas de la
edición v3.1 de `objective_variants_cost`. Mide por separado:

- la actualización de la sombra de F2 tras un paso, sobre todos los parámetros ajustables;
- el intercambio de pesos de cada validación con F2 o F3, con los dos sellados;
- el cálculo fuera de sitio del punto x de F3, con un z fijado a mano;
- la contracción de F3, que construye un predictor nuevo una sola vez por ejecución;
- la inferencia sin grafo de T instantes, como referencia de lo que cuesta una validación;
- los bytes que F2 y F3 añaden a cada estado guardado.

Ningún parámetro del predictor medido cambia, y se comprueba al final.
"""

import argparse
import hashlib
import os
import platform
import resource
import statistics
import subprocess
import time
from dataclasses import replace
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from benchmarks.objective_variants_cost import BASE, batches, inference, specification
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.training.financial_run import parameter_roles
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy
from mars_titan.training.schedule_free import AveragedPoint, ScheduleFree
from mars_titan.training.shrink_perturb import ShrinkPerturb, shrink_perturb
from mars_titan.training.weight_averaging import ShadowWeights, WeightAverage, applied

SOURCES = (
    "benchmarks/weight_options_cost.py",
    "src/mars_titan/training/weight_averaging.py",
    "src/mars_titan/training/schedule_free.py",
    "src/mars_titan/training/shrink_perturb.py",
)


def _commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _summary(values):
    return dict(
        median=statistics.median(values), minimum=min(values), maximum=max(values), samples=values
    )


def _timed(function, repetitions, warmup):
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        samples.append(time.perf_counter() - started)
    return _summary(samples)


def _digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _bytes(tensors):
    # El estado del modelo incluye también entradas que no son tensores, como su contrato.
    return sum(
        value.numel() * value.element_size() for value in tensors if isinstance(value, torch.Tensor)
    )


def measure(rows, instants, repetitions, warmup):
    spec = specification()
    model = FinancialPredictor(FinancialConfig(spec, variant="mac_online", **BASE))
    roles = parameter_roles(model)
    named = dict(model.named_parameters())
    trainable = {name: named[name] for names in roles.values() for name in names}
    before = _digest(model)
    shadow = ShadowWeights(WeightAverage(decay=0.99), trainable)
    groups = [
        dict(params=[named[name] for name in names], role=role)
        for role, names in roles.items()
        if names
    ]
    optimizer = ScheduleFree().build(groups, learning_rate=1e-3, weight_decay=0.0)
    for value in trainable.values():
        # z fijado a mano para que x difiera de y, sin ningún paso.
        optimizer.state[value]["z"] = value.detach() + 1e-3
    point = AveragedPoint(optimizer, trainable)

    def swap(source):
        def run():
            with applied(named, source.emitted()):
                model._seal_parameters()
            model._seal_parameters()

        return run

    def shrink():
        fresh = FinancialPredictor(replace(model.config, seed=1))
        copy = {name: value.detach().clone() for name, value in trainable.items()}
        shrink_perturb(copy, dict(fresh.named_parameters()), ShrinkPerturb())

    inputs = batches(spec, rows, instants)
    with torch.no_grad():
        results = dict(
            weight_ema_update=_timed(lambda: shadow.update(trainable), 20 * repetitions, warmup),
            weight_ema_validation_swap=_timed(swap(shadow), repetitions, warmup),
            schedule_free_point=_timed(point.emitted, repetitions, warmup),
            schedule_free_validation_swap=_timed(swap(point), repetitions, warmup),
            shrink_perturb_once=_timed(shrink, max(1, repetitions // 2), min(warmup, 1)),
        )
    results["inference_reference"] = _timed(lambda: inference(model, inputs), repetitions, warmup)
    if _digest(model) != before:
        raise ValueError("La medida modificó los parámetros del predictor")
    shadow_bytes = _bytes(shadow.shadow)
    model_bytes = _bytes(model.state_dict().values())
    return dict(
        timings=results,
        parameters=dict(
            trainable=sum(value.numel() for value in trainable.values()),
            total=sum(value.numel() for value in model.parameters()),
        ),
        memory=dict(
            weight_ema_shadow_bytes=shadow_bytes,
            model_state_bytes=model_bytes,
            weight_ema_extra_state_bytes=model_bytes + shadow_bytes,
            schedule_free_extra_state_bytes=model_bytes,
            note=(
                "F2 añade al estado los pesos del ajuste y las sombras. F3 añade los pesos del "
                "ajuste. El estado del optimizador sin calendario guarda z y el segundo "
                "momento, tantos tensores como los dos momentos de AdamW."
            ),
        ),
        parameters_unchanged=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=256)
    parser.add_argument("--instants", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    torch.set_num_threads(arguments.threads)
    policy = apply_kernel_policy(FP32_STRICT)
    load_start = os.getloadavg()
    started = time.perf_counter()
    with sdpa_kernel(SDPBackend.MATH):
        measured = measure(
            arguments.rows, arguments.instants, arguments.repetitions, arguments.warmup
        )
    atomic_json(
        arguments.output,
        dict(
            schema_version=1,
            kind="weight_options_cost",
            issue=749,
            recorded_at_utc=datetime.now(UTC).isoformat(),
            commit=_commit(),
            scope="cpu_without_optimizer_steps_or_training",
            device="cpu",
            cpu=cpu_name(),
            versions=dict(
                python=platform.python_version(),
                torch=torch.__version__,
                schedulefree=metadata.version("schedulefree"),
            ),
            kernel_policy=policy,
            sdpa_backend="MATH",
            source_sha256={
                path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in SOURCES
            },
            threads=arguments.threads,
            shapes=dict(rows=arguments.rows, instants=arguments.instants),
            base_predictor=BASE,
            repetitions=arguments.repetitions,
            warmup=arguments.warmup,
            optimizer_steps=0,
            training_runs=0,
            total_wall_seconds=time.perf_counter() - started,
            process_peak_ram_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            load_average_start=list(load_start),
            load_average_end=list(os.getloadavg()),
            **measured,
        ),
    )


if __name__ == "__main__":
    main()
