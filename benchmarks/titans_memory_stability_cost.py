"""Medir el coste de PT1 con un forward y un backward sobre entradas reales, sin optimizador.

La medida tiene dos fases para no ocupar la GPU mientras se leen datos. `prepare` se ejecuta
sin GPU, abre una vista walk-forward, prepara o reutiliza el índice de observaciones del
tramo de ajuste y guarda las entradas de unos pocos eventos consecutivos para un conjunto
fijo de flujos. No guarda etiquetas ni objetivos. `measure` carga ese archivo y, para cada
brazo de la declaración de PT1, recorre los eventos con el grafo completo, calcula la
pérdida pinball contra un objetivo nulo y ejecuta un backward. Los gradientes se descartan
y se comprueba al final que ningún parámetro ha cambiado. El objetivo nulo solo da forma a
la pérdida, porque el coste del backward no depende de su valor.

Solo se registran tiempos, memoria del asignador y la fracción de filas recortadas al
inicio, nunca pérdidas ni errores, así que la medida no es una evaluación.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES
from mars_titan.data.storage import atomic_json
from mars_titan.models.quantile_head import pinball_loss
from mars_titan.models.titans import neural_memory
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.financial_inputs import DecisionBatch, FinancialInputSpec

DAY_US = 86_400_000_000


def _commit():
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(view, work, recipe, rows, events, output):
    """Guardar las entradas de `events` eventos para `rows` flujos presentes en todos ellos."""
    from mars_titan.training import titans_walk_forward as titans
    from mars_titan.training.campaign_throughput import _view_fold
    from mars_titan.training.corpus_inputs import CorpusDataset

    if torch.cuda.is_available():
        raise RuntimeError("La fase de lectura se ejecuta sin GPU, con CUDA_VISIBLE_DEVICES=-1")
    started = time.perf_counter()
    document = json.loads(Path(recipe).read_text())
    dataset = CorpusDataset(Path(view), input_policy=HISTORICAL_MASKED)
    fold = _view_fold(dataset)
    warmup = titans.walk_forward_options(document)["warmup_months"]
    phase = titans.window_phases(fold, warmup)["train"]
    source = titans._sources(dataset, {"train": phase}, Path(work) / "titans-indices")["train"]
    specification = source.specification()
    collected = []
    for event in source.batched_events(block_rows=256):
        if not event.inputs:
            continue
        by_flow = {}
        for block in event.inputs:
            for index, sample in enumerate(block["sample_ids"]):
                by_flow[sample.rsplit("/", 1)[0]] = (block, index)
        collected.append((event.at, by_flow))
        if len(collected) == events:
            break
    if len(collected) < events:
        raise ValueError("El tramo de ajuste no tiene eventos suficientes")
    common = sorted(set.intersection(*(set(flows) for _, flows in collected)))
    if len(common) < rows:
        raise ValueError("No hay flujos suficientes presentes en todos los eventos")
    flows = common[:rows]
    arrays = {}
    for name in MODALITIES:
        arrays[name] = np.stack(
            [
                np.stack([block["inputs"][name][index] for block, index in (f[x] for x in flows)])
                for _, f in collected
            ]
        )
    arrays["presence"] = np.stack(
        [
            np.stack([block["presence"][index] for block, index in (f[x] for x in flows)])
            for _, f in collected
        ]
    )
    arrays["input_available_at"] = np.stack(
        [
            np.stack(
                [block["input_available_at"][index] for block, index in (f[x] for x in flows)]
            ).astype("datetime64[us]")
            for _, f in collected
        ]
    )
    arrays["prediction_at"] = np.array([at for at, _ in collected], dtype=np.int64)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    metadata = dict(
        schema_version=1,
        kind="titans_memory_stability_cost_inputs",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        view=str(view),
        view_identity=dataset.identity,
        fold=fold,
        phase=asdict(phase),
        flows=flows,
        events=events,
        rows=rows,
        specification=dict(
            source_sha256=specification.source_sha256,
            view_sha256=specification.view_sha256,
            representation=specification.representation,
            dimensions=specification.dimensions,
            input_policy=specification.input_policy,
        ),
        labels_saved=False,
        inputs_sha256=_sha256(output),
        seconds=time.perf_counter() - started,
    )
    atomic_json(output.with_suffix(".json"), metadata)
    print(json.dumps({key: metadata[key] for key in ("rows", "events", "seconds")}))


def _batches(arrays, metadata, specification, device):
    flows = metadata["flows"]
    result = []
    for event, at in enumerate(arrays["prediction_at"].tolist()):
        raw = dict(
            inputs={name: arrays[name][event] for name in MODALITIES},
            presence=arrays["presence"][event],
            sample_ids=[f"{flow}/{at}" for flow in flows],
            market=[flow.split("/", 1)[0] for flow in flows],
            prediction_at=np.full(len(flows), at, dtype="datetime64[us]"),
            input_available_at=arrays["input_available_at"][event],
        )
        result.append(
            DecisionBatch.from_corpus(raw, specification, device=device, dtype=torch.float32)
        )
    return result


def _parameters_digest(model):
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _pass(model, batches, flows, device):
    """Un forward con grafo por los eventos y un backward, sin aplicar ningún gradiente."""
    synchronize = torch.cuda.synchronize if device.type == "cuda" else (lambda *_: None)
    if device.type == "cuda":
        synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    state = model.initial_state(flows, differentiable=True)
    outputs = []
    for batch in batches:
        prepared = model.prepare(batch, state, differentiable=True)
        state = prepared.next_state
        outputs.append(prepared.quantiles)
    quantiles = torch.cat(outputs)
    loss = pinball_loss(quantiles, torch.zeros(len(quantiles), device=device))
    synchronize(device)
    forward = time.perf_counter() - started
    started = time.perf_counter()
    loss.backward()
    synchronize(device)
    backward = time.perf_counter() - started
    for value in model.parameters():
        value.grad = None
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    return forward, backward, peak


def _clipped_fraction(model, batches, flows):
    """Fracción de filas que PT1 recorta en el primer recorrido, con los pesos iniciales."""
    counts = [0, 0]
    original = neural_memory.clip_rows

    def counting(gradient, limit):
        squared = gradient.detach().square().sum(dim=-1)
        counts[0] += int((squared > limit * limit).sum())
        counts[1] += squared.numel()
        return original(gradient, limit)

    neural_memory.clip_rows = counting
    try:
        with torch.no_grad():
            state = model.initial_state(flows)
            for batch in batches:
                state = model.prepare(batch, state).next_state
    finally:
        neural_memory.clip_rows = original
    return counts[0] / counts[1] if counts[1] else None


def measure(inputs, recipe, declaration, device, repeats, output):
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Se pidió cuda:0 y no hay dispositivo CUDA visible")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(2)
    inputs = Path(inputs)
    metadata = json.loads(inputs.with_suffix(".json").read_text())
    if _sha256(inputs) != metadata["inputs_sha256"]:
        raise ValueError("Las entradas guardadas no coinciden con su huella")
    arrays = dict(np.load(inputs))
    specification = FinancialInputSpec(**metadata["specification"])
    document = json.loads(Path(recipe).read_text())
    arms = json.loads(Path(declaration).read_text())["arms"]
    base = {key: value for key, value in document["predictor"].items() if key != "dtype"}
    batches = _batches(arrays, metadata, specification, device)
    flows = tuple(metadata["flows"])
    gpu = None
    if device.type == "cuda":
        gpu = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    results = {}
    for arm, record in arms.items():
        options = base | record["predictor_changes"]
        config = FinancialConfig(specification, variant="mac_online", seed=42, **options)
        model = FinancialPredictor(config, device=device, dtype=torch.float32).train()
        before = _parameters_digest(model)
        measures = [_pass(model, batches, flows, device) for _ in range(repeats + 1)][1:]
        forward, backward, peaks = zip(*measures, strict=True)
        clipped = (
            _clipped_fraction(model, batches, flows)
            if config.memory_stability is not None and config.memory_stability.gradient_clip
            else None
        )
        if _parameters_digest(model) != before:
            raise RuntimeError("La medida ha modificado parámetros")
        results[arm] = dict(
            role=record["role"],
            predictor_changes=record["predictor_changes"],
            forward_seconds=list(forward),
            backward_seconds=list(backward),
            forward_median=float(np.median(forward)),
            backward_median=float(np.median(backward)),
            peak_allocated_bytes=list(peaks) if device.type == "cuda" else None,
            clipped_row_fraction_at_initialization=clipped,
            parameters_unchanged=True,
        )
        print(
            arm, round(results[arm]["forward_median"], 4), round(results[arm]["backward_median"], 4)
        )
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    control = next(name for name, value in results.items() if value["role"] == "control")
    for value in results.values():
        total = value["forward_median"] + value["backward_median"]
        reference = results[control]["forward_median"] + results[control]["backward_median"]
        value["relative_time_vs_control"] = total / reference
    receipt = dict(
        schema_version=1,
        kind="titans_memory_stability_cost",
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=_commit(),
        scope=(
            "Un forward con grafo por eventos reales y un backward por repetición, sin "
            "optimizador ni pasos. Objetivo nulo solo para dar forma a la pérdida."
        ),
        device=str(device),
        gpu=gpu,
        versions=dict(python=platform.python_version(), torch=torch.__version__),
        precision="float32 sin TF32",
        threads=torch.get_num_threads(),
        recipe=str(recipe),
        recipe_sha256=_sha256(recipe),
        declaration=str(declaration),
        declaration_sha256=_sha256(declaration),
        inputs=dict(
            sha256=metadata["inputs_sha256"],
            view_identity=metadata["view_identity"],
            fold=metadata["fold"],
            rows=metadata["rows"],
            events=metadata["events"],
            labels_saved=metadata["labels_saved"],
        ),
        repeats=repeats,
        warmup_passes=1,
        arms=results,
        optimizer_steps=0,
        training_runs=0,
        evaluation=False,
    )
    atomic_json(Path(output), receipt)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    read = commands.add_parser("prepare")
    read.add_argument("--view", type=Path, required=True)
    read.add_argument("--work", type=Path, required=True)
    read.add_argument("--recipe", type=Path, required=True)
    read.add_argument("--rows", type=int, default=128)
    read.add_argument("--events", type=int, default=8)
    read.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("measure")
    run.add_argument("--inputs", type=Path, required=True)
    run.add_argument("--recipe", type=Path, required=True)
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--device", default="cuda:0", choices=("cpu", "cuda:0"))
    run.add_argument("--repeats", type=int, default=5)
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.view, args.work, args.recipe, args.rows, args.events, args.output)
    else:
        measure(args.inputs, args.recipe, args.declaration, args.device, args.repeats, args.output)


if __name__ == "__main__":
    main()
