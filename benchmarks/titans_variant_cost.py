"""Medir en CPU el coste de los modelos de una declaración de Titans, sin optimizador.

Reutiliza el archivo que `titans_paper_projections_cost.py materialize` lee del tramo de
ajuste de una vista de la campaña, con sus bits de presencia reales y sin objetivos. Para
cada modelo de la declaración (por ejemplo PT4, PT5 o PT6) recorre los T instantes con
grafo, calcula la pinball frente a ceros y ejecuta un backward. Los gradientes se descartan
en cada repetición y al final se comprueba que ningún parámetro ha cambiado. Mide también la
inferencia sin grafo y cuenta las filas con alguna modalidad ausente. Las repeticiones se
alternan entre modelos, así que todos sufren la misma deriva de la máquina. Con el contexto
de PT4, cada instante se lee en un bloque y se confirma después de emitirlo, como en el
recorrido.
"""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from benchmarks.titans_paper_projections_cost import _raw_batches
from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch, validated_cpu_batch


def _commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


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
        values=ordered,
    )


def _step(model, batch, state, context, *, differentiable):
    """Una decisión por instante. Con PT4, la sesión se confirma tras emitirla."""
    options = {} if context is None else dict(context=context)
    prepared = model.prepare(batch, state, differentiable=differentiable, **options)
    if context is not None:
        context = model.commit_context(prepared.next_context, differentiable=differentiable)
    return prepared, context


def _initial_context(model, *, differentiable):
    if model.session_context is None:
        return None
    return model.initial_context(differentiable=differentiable)


def _context_bytes(model, flows, batches):
    """Bytes de tensores del contexto compartido tras recorrer los instantes, o cero."""
    context = _initial_context(model, differentiable=False)
    if context is None:
        return 0
    state = model.initial_state(flows)
    with torch.no_grad():
        for batch in batches:
            prepared, context = _step(model, batch, state, context, differentiable=False)
            state = prepared.next_state
    payload = model.export_context(context)
    tensors = [payload["means"]] if payload["means"] is not None else []
    for memory in payload["memory"] or []:
        tensors.extend(
            value for key in ("weights", "momentum", "levels") for value in memory.get(key, ())
        )
    return sum(value.numel() * value.element_size() for value in tensors)


def _graph_pass(model, batches, flows):
    started = time.perf_counter()
    state = model.initial_state(flows, differentiable=True)
    context = _initial_context(model, differentiable=True)
    outputs = []
    for batch in batches:
        prepared, context = _step(model, batch, state, context, differentiable=True)
        state = prepared.next_state
        outputs.append(prepared.quantiles)
    quantiles = torch.cat(outputs)
    loss = pinball_loss(quantiles, torch.zeros(len(quantiles), dtype=quantiles.dtype))
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    backward = time.perf_counter() - started
    for value in model.parameters():
        value.grad = None
    return forward, backward


def _inference(model, batches, flows):
    started = time.perf_counter()
    with torch.no_grad():
        state = model.initial_state(flows)
        context = _initial_context(model, differentiable=False)
        for batch in batches:
            prepared, context = _step(model, batch, state, context, differentiable=False)
            state = prepared.next_state
    return time.perf_counter() - started


def measure(data, declaration_path, output, *, repetitions, warmup, threads, sector_source=None):
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    metadata, spec, raws = _raw_batches(data)
    declaration_path = Path(declaration_path)
    declaration = json.loads(declaration_path.read_text())
    recipe = Path(declaration["base_recipe"])
    document = json.loads(recipe.read_text())
    if document["predictor"]["dtype"] != "float32":
        raise ValueError("La medida reproduce la receta FP32")
    base = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    batches = [
        DecisionBatch.from_validated(validated_cpu_batch(raw, spec), dtype=torch.float32)
        for raw in raws
    ]
    flows = batches[0].flow_ids
    presence = np.concatenate([np.asarray(raw["presence"], dtype=bool) for raw in raws])
    incomplete = ~presence.all(axis=1)
    built = {}
    for name, arm in declaration["arms"].items():
        changes = dict(arm["predictor_changes"])
        if "sector_map" in changes:
            # La ficha de sectores se lee de la copia local del conjunto, que no se versiona.
            if sector_source is None:
                raise ValueError("El control sectorial necesita --sector-source")
            changes["sector_map"] = dict(source=str(sector_source))
        config = FinancialConfig(spec, variant=declaration["variant"], seed=42, **(base | changes))
        model = FinancialPredictor(config, dtype=torch.float32).train()
        built[name] = (arm, model, _digest(model))
    for _ in range(warmup):
        for _, model, _ in built.values():
            _graph_pass(model, batches, flows)
            _inference(model, batches, flows)
    # Las repeticiones se alternan entre modelos y rotan su orden, para que la deriva de la
    # máquina (frecuencia, caché, otras cargas) no favorezca al que se mide primero.
    graph, inference = {name: [] for name in built}, {name: [] for name in built}
    names = list(built)
    for repetition in range(repetitions):
        shift = repetition % len(names)
        for name in names[shift:] + names[:shift]:
            model = built[name][1]
            graph[name].append(_graph_pass(model, batches, flows))
            inference[name].append(_inference(model, batches, flows))
    models = {}
    for name, (arm, model, before) in built.items():
        if _digest(model) != before:
            raise RuntimeError("La medida ha modificado parámetros")
        flows_state = model.initial_state(flows)
        models[name] = dict(
            role=arm["role"],
            predictor_changes=arm["predictor_changes"],
            total_parameters=sum(value.numel() for value in model.parameters()),
            state_tensor_bytes_per_flow=model.state_usage(flows_state)["tensor_bytes_per_flow"],
            shared_context_tensor_bytes=_context_bytes(model, flows, batches),
            session_context=model.config.identity().get("session_context"),
            forward_with_graph_seconds=_summary([value[0] for value in graph[name]]),
            backward_seconds=_summary([value[1] for value in graph[name]]),
            inference_seconds=_summary(inference[name]),
            parameters_unchanged=True,
        )
        print(name, round(models[name]["forward_with_graph_seconds"]["median"], 4), flush=True)
    control = next(value for value in models.values() if value["role"] == "control")
    for value in models.values():
        value["added_parameters"] = value["total_parameters"] - control["total_parameters"]
        value["relative_to_control"] = {
            key: value[key]["median"] / control[key]["median"]
            for key in ("forward_with_graph_seconds", "backward_seconds", "inference_seconds")
        }
    receipt = dict(
        schema_version=1,
        kind="titans_variant_cost",
        innovation=declaration["innovation"],
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Forward con grafo, pinball frente a ceros y backward, más inferencia sin grafo, "
            "sobre entradas reales sin objetivos. Sin optimizador ni pasos."
        ),
        device="cpu",
        cpu=platform.processor() or platform.machine(),
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        precision="float32",
        threads=torch.get_num_threads(),
        declaration=str(declaration_path),
        declaration_sha256=sha256(declaration_path),
        recipe=str(recipe),
        recipe_sha256=sha256(recipe),
        inputs=dict(
            data_sha256=metadata["data_sha256"],
            view_manifest_sha256=metadata["view_manifest_sha256"],
            flows=len(flows),
            instants=len(batches),
            dimensions=metadata["dimensions"],
            targets_read=metadata.get("targets_read", False),
            rows=int(len(presence)),
            rows_with_some_absence=int(incomplete.sum()),
            absent_rows_by_modality={
                name: int((~presence[:, index]).sum()) for index, name in enumerate(MODALITIES)
            },
        ),
        repetitions=repetitions,
        warmup=warmup,
        order="repeticiones alternadas entre modelos con orden rotatorio",
        models=models,
        optimizer_steps=0,
        training_runs=0,
        evaluation=False,
    )
    atomic_json(Path(output), receipt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--declaration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--sector-source", type=Path)
    args = parser.parse_args()
    if torch.cuda.is_available():
        raise RuntimeError("La medida se hace en CPU, con CUDA_VISIBLE_DEVICES=-1")
    measure(
        args.data,
        args.declaration,
        args.output,
        repetitions=args.repetitions,
        warmup=args.warmup,
        threads=args.threads,
        sector_source=args.sector_source,
    )


if __name__ == "__main__":
    main()
