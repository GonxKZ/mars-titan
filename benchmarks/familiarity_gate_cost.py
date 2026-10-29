"""Medir el coste de R3 de #786 frente al lector episódico de referencia, sin optimizador.

El lector M1 con K = 1 lee una instantánea de episodios con claves unitarias al azar, con las
formas de la receta del lector: bloques de 512 filas, 64 coordenadas, ocho vecinos y 1.024
episodios. Cada modelo ejecuta la lectura con grafo, un error absoluto frente a objetivos
sintéticos y su backward, y aparte la lectura sin grafo. Las repeticiones alternan los modelos
con orden rotatorio. Antes de medir se comprueba que cada puerta nueva reproduce bit a bit el
lector de referencia y al terminar que ningún parámetro ha cambiado. No hay optimizador.
"""

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

from benchmarks.objective_variants_cost import _summary
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.models.titans import episodic_readout
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.training.kernel_policy import FP32_STRICT, apply_kernel_policy

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = "configs/experiments/familiarity-gate-v1.json"
CODEC, CONTEXT = "a" * 64, "b" * 64
HIDDEN, NEIGHBORS, CUTOFF = 64, 8, 10


def arms():
    """El lector de referencia y los modelos de la declaración, en ese orden."""
    declaration = json.loads((ROOT / DECLARATION).read_text(encoding="utf-8"))
    result = {"reference": None}
    for name, arm in declaration["arms"].items():
        result[name] = arm["recipe_changes"]["familiarity_gate"]
    return result


def _digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _synchronize(device):
    if device == "cuda:0":
        torch.cuda.synchronize(device)


def snapshot(episodes, device):
    generator = torch.Generator().manual_seed(5)
    keys = torch.randn(episodes, 64, generator=generator, dtype=torch.float64)
    keys = keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)
    return EpisodeSnapshot.create(
        keys=keys.float(),
        values=torch.randn(episodes, 64, generator=generator),
        labels=torch.randn(episodes, generator=generator, dtype=torch.float64) * 0.015,
        ids=torch.arange(1, episodes + 1, dtype=torch.int64),
        decision_at=torch.ones(episodes, dtype=torch.int64),
        available_at=torch.ones(episodes, dtype=torch.int64),
        maturity_at=torch.full((episodes,), 2, dtype=torch.int64),
        cutoff=CUTOFF,
        codec_id=CODEC,
        context_id=CONTEXT,
        device=device,
        dtype=torch.float32,
    )


def reader(gate, device):
    config = EpisodicReadoutConfig(
        CODEC,
        hidden_size=HIDDEN,
        neighbors=NEIGHBORS,
        seed=42,
        max_batch=1024,
        max_working_bytes=128 * 1024**2,
        familiarity_gate=gate,
    )
    return EpisodicReadout(config, dtype=torch.float32, device=device).train()


def graph_pass(model, data, state, head, target):
    device = str(state.device)
    _synchronize(device)
    started = time.perf_counter()
    result = model(state, data, context_id=CONTEXT, cutoff=CUTOFF, differentiable=True)
    loss = torch.nn.functional.l1_loss(result.state @ head, target)
    _synchronize(device)
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    _synchronize(device)
    backward = time.perf_counter() - started
    if not all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()):
        raise ValueError("La medida encontró un gradiente no finito")
    for value in model.parameters():
        value.grad = None
    return forward, backward


def inference(model, data, state):
    device = str(state.device)
    _synchronize(device)
    started = time.perf_counter()
    with torch.no_grad():
        model(state, data, context_id=CONTEXT, cutoff=CUTOFF)
    _synchronize(device)
    return time.perf_counter() - started


def measure(output, *, rows, episodes, repetitions, warmup, threads, device="cpu"):
    limits = (
        (rows, 1, 1024),
        (episodes, NEIGHBORS, 1024),
        (repetitions, 1, 50),
        (warmup, 0, 10),
        (threads, 1, 8),
    )
    if any(type(value) is not int or not low <= value <= high for value, low, high in limits):
        raise ValueError("Las formas, repeticiones o hilos exceden el presupuesto de la medida")
    if device not in ("cpu", "cuda:0") or (device == "cuda:0" and not torch.cuda.is_available()):
        raise ValueError("La medida usa cpu o cuda:0 disponible, sin cambiar de dispositivo")
    if Path(output).exists():
        raise FileExistsError("La medida necesita un recibo nuevo")
    started, load_at_start = time.perf_counter(), os.getloadavg()
    torch.set_num_threads(threads)
    policy = apply_kernel_policy(FP32_STRICT)
    generator = torch.Generator().manual_seed(11)
    state = torch.randn(rows, HIDDEN, generator=generator).to(device)
    head = (torch.randn(HIDDEN, generator=generator) / HIDDEN**0.5).to(device)
    target = (torch.randn(rows, generator=generator) * 0.015).to(device)
    data = snapshot(episodes, device)
    declared = arms()
    built = {name: reader(gate, device) for name, gate in declared.items()}
    digests = {name: _digest(model) for name, model in built.items()}
    with torch.no_grad():
        expected = built["reference"](state, data, context_id=CONTEXT, cutoff=CUTOFF).state
        for name, model in built.items():
            # Con a = b = 0 cada puerta vale uno y el estado es el de la referencia.
            if not torch.equal(
                model(state, data, context_id=CONTEXT, cutoff=CUTOFF).state, expected
            ):
                raise RuntimeError(f"{name} no empieza en el lector de referencia")
    names = list(built)
    for _ in range(warmup):
        for model in built.values():
            graph_pass(model, data, state, head, target)
            inference(model, data, state)
    graph = {name: [] for name in names}
    plain = {name: [] for name in names}
    for repetition in range(repetitions):
        shift = repetition % len(names)
        for name in names[shift:] + names[:shift]:
            graph[name].append(graph_pass(built[name], data, state, head, target))
            plain[name].append(inference(built[name], data, state))
    models = {}
    for name, model in built.items():
        if _digest(model) != digests[name]:
            raise RuntimeError("La medida ha modificado parámetros")
        models[name] = dict(
            familiarity_gate=declared[name],
            total_parameters=sum(value.numel() for value in model.parameters()),
            estimated_bytes=model.estimated_bytes(rows, episodes),
            estimated_bytes_with_graph=model.estimated_bytes(rows, episodes, differentiable=True),
            forward_with_graph_seconds=_summary([value[0] for value in graph[name]]),
            backward_seconds=_summary([value[1] for value in graph[name]]),
            inference_seconds=_summary(plain[name]),
            starts_at_reference=True,
            parameters_unchanged=True,
        )
    control = models["reference"]
    for value in models.values():
        value["added_parameters"] = value["total_parameters"] - control["total_parameters"]
        value["relative_to_reference"] = {
            key: value[key]["median"] / control[key]["median"]
            for key in ("forward_with_graph_seconds", "backward_seconds", "inference_seconds")
        }
    sources = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in (
            ("episodic_readout", Path(episodic_readout.__file__)),
            ("benchmark", Path(__file__)),
            ("declaration", ROOT / DECLARATION),
        )
    }
    receipt = dict(
        schema_version=1,
        kind="familiarity_gate_cost",
        issue=786,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        scope=(
            "Lectura del lector M1 con K = 1 con grafo, error absoluto y backward, más la "
            "lectura sin grafo, sobre una instantánea sintética. Sin optimizador ni pasos."
        ),
        device=device,
        cpu=cpu_name(),
        gpu=torch.cuda.get_device_name(0) if device == "cuda:0" else None,
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        precision="float32",
        kernel_policy=policy,
        threads=threads,
        shape=dict(rows=rows, episodes=episodes, hidden_size=HIDDEN, neighbors=NEIGHBORS),
        repetitions=repetitions,
        warmup=warmup,
        load_average=dict(start=list(load_at_start), end=list(os.getloadavg())),
        sources_sha256=sources,
        models=models,
        elapsed_seconds=time.perf_counter() - started,
    )
    atomic_json(Path(output), receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, default=512)
    parser.add_argument("--episodes", type=int, default=1024)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    arguments = parser.parse_args()
    receipt = measure(
        arguments.output,
        rows=arguments.rows,
        episodes=arguments.episodes,
        repetitions=arguments.repetitions,
        warmup=arguments.warmup,
        threads=arguments.threads,
        device=arguments.device,
    )
    for name, value in receipt["models"].items():
        print(name, json.dumps(value["relative_to_reference"]))


if __name__ == "__main__":
    main()
