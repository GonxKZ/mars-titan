"""Caudal de las referencias neuronales de la campaña A sobre lotes reales, sin aprender.

La medida tiene dos fases para no ocupar la GPU mientras se lee el corpus:

1. `materialize`, sin GPU: lee una vez lotes reales de ajuste de una vista, cronometra el
   lector y los guarda en un archivo compacto.
2. `measure`, con GPU: carga ese archivo y mide por caso el paso forward, pinball y
   backward sin optimizador (los gradientes se liberan sin tocar pesos) y la inferencia
   sin gradiente con los lotes residentes, el mismo paso copiando cada lote desde la
   memoria del proceso como el ajuste, las sincronizaciones de un paso del bucle de
   `reference_run` hasta el optimizador y la memoria de la GPU.

Para contrastar dos versiones del código basta con cambiar `PYTHONPATH` y alternar las
ejecuciones de `measure` sobre el mismo archivo.

    python benchmarks/reference_kernels.py materialize VIEW --output lotes.pt
    PYTHONPATH=src python benchmarks/reference_kernels.py measure lotes.pt --output r.json
"""

import argparse
import inspect
import json
import math
import os
import platform
import resource
import statistics
import subprocess
import time
import warnings
from pathlib import Path

import torch

from mars_titan.budget_training import seed_run
from mars_titan.models.baselines import transformer as transformer_module
from mars_titan.models.baselines.multimodal import PRESENCE_FUSION, MultimodalReference
from mars_titan.models.quantile_head import QUANTILE_HEAD, pinball_loss
from mars_titan.training import reference_run

INPUT_POLICY = "historical_masked_2000_v1"
TRANSFORMER = dict(heads=4, feedforward_multiplier=2)


def reference_case(kind, hidden_size, layers, dropout):
    architecture = dict(hidden_size=hidden_size, layers=layers, dropout=dropout)
    if kind == "transformer":
        architecture["transformer"] = dict(TRANSFORMER)
    return dict(kind=kind, seed=42, architecture=architecture)


# Candidatos -00 y -10 de la campaña A, escritos aquí para que las dos versiones del código
# midan los mismos casos. El Transformer -11 añade el camino de dos capas del diseño.
CASES = [
    (f"{kind}-{index}", reference_case(kind, *architecture))
    for kind in ("rnn", "lstm", "gru", "dlinear", "transformer")
    for index, architecture in (("00", (32, 1, 0.0)), ("10", (64, 1, 0.2)))
] + [("transformer-11", reference_case("transformer", 128, 2, 0.2))]


def batch_options(kind, size):
    """Lote del Transformer en ambas versiones: opción explícita o límite ampliado."""
    if "max_batch" in inspect.signature(MultimodalReference).parameters:
        from mars_titan.models.baselines.multimodal import transformer_batch_options

        return transformer_batch_options(kind, size)
    # Versión anterior: el lote estaba fijado en el módulo. Solo en este proceso.
    transformer_module._MAX_BATCH = 1 << 20
    transformer_module._MAX_ATTENTION_ELEMENTS = 1 << 40
    transformer_module.CompactPriceTransformer.max_batch = 1 << 20
    return {}


def build(case, dimensions, context, size):
    seed_run(case["seed"])
    return MultimodalReference(
        case["kind"],
        dimensions,
        context=context,
        mask_fusion=PRESENCE_FUSION,
        head=QUANTILE_HEAD,
        **case["architecture"],
        **batch_options(case["kind"], size),
    )


def resident(batch, device):
    return dict(
        inputs={k: torch.from_numpy(v).to(device) for k, v in batch["inputs"].items()},
        presence=torch.from_numpy(batch["presence"]).to(device),
        target=torch.from_numpy(batch["target"]).to(device, dtype=torch.float32),
    )


def train_step(model, batch):
    emitted = model(batch["inputs"], batch["presence"])
    pinball_loss(emitted, batch["target"]).backward()
    # Sin optimizador: los gradientes se liberan sin tocar ningún peso.
    model.zero_grad(set_to_none=True)


def timed(function, batches, *, warmup, repeats):
    for batch in batches[:warmup]:
        function(batch)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for batch in batches[warmup:]:
            function(batch)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) / len(batches[warmup:]))
    return values


def summary(values, size):
    median = statistics.median(values)
    return dict(
        ms_per_batch=median * 1e3,
        ms_min=min(values) * 1e3,
        ms_max=max(values) * 1e3,
        samples_per_second=size / median,
    )


def copied_rate(model, batches, *, size, warmup, device):
    """Paso con la copia de cada lote desde la memoria del proceso, como en el ajuste."""
    model.train()
    for index, batch in enumerate(batches):
        if index == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        emitted = reference_run._forward(model, batch, device)
        target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
        pinball_loss(emitted, target).backward()
        model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return size * (len(batches) - warmup) / (time.perf_counter() - start)


def apply_precision(name):
    """FP32 estricto como `kernel_policy.apply_kernel_policy`, o los valores de PyTorch."""
    strict = name == "strict"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = not strict
    torch.set_float32_matmul_precision("highest")


def process_mib():
    """Memoria del proceso según nvidia-smi: contexto CUDA más lo reservado por PyTorch."""
    lines = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()
    for line in lines:
        pid, used = (value.strip() for value in line.split(","))
        if int(pid) == os.getpid():
            return float(used)
    return None


def count_syncs(function):
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()
    try:
        torch.cuda.set_sync_debug_mode(1)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            function()
    finally:
        torch.cuda.set_sync_debug_mode(previous)
    return sum("synchroniz" in str(item.message) for item in caught)


def loop_step(model, batch, device):
    """Paso del bucle de `reference_run` hasta el optimizador, que aquí no se llama."""
    statistics_ = dict(samples=0, squared_error=0.0, absolute_error=0.0)
    target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
    emitted = reference_run._forward(model, batch, device)
    prediction = reference_run._point(emitted, target, True)
    loss = pinball_loss(emitted, target, reduction="none").mean()
    if not torch.isfinite(loss).item():
        raise ValueError("La pérdida no es finita")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
    reference_run._update(statistics_, prediction, torch.from_numpy(batch["target"]).to(device))
    model.zero_grad(set_to_none=True)


def materialize(args):
    """Leer lotes reales sin GPU, cronometrar el lector y guardarlos."""
    dataset = reference_run.configured_corpus(args.view, input_policy=INPUT_POLICY)
    start = time.perf_counter()
    loaded, first = [], None
    for batch in dataset.batches(
        partition="train", batch_size=args.batch_size, epoch=0, seed=args.seed
    ):
        loaded.append(batch)
        first = first or time.perf_counter() - start
        if len(loaded) == args.batches:
            break
    elapsed = time.perf_counter() - start
    torch.save(
        dict(
            view=str(args.view),
            input_policy=INPUT_POLICY,
            context=dataset.context,
            batch_size=args.batch_size,
            seed=args.seed,
            loader=dict(
                first_batch_seconds=first,
                later_batches_samples_per_second=args.batch_size
                * (len(loaded) - 1)
                / (elapsed - first),
                max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            ),
            batches=[
                dict(
                    inputs={k: torch.from_numpy(v) for k, v in batch["inputs"].items()},
                    presence=torch.from_numpy(batch["presence"]),
                    target=torch.from_numpy(batch["target"]),
                )
                for batch in loaded
            ],
        ),
        args.output,
    )


def stored_batches(path, size):
    """Lotes guardados como arrays, divididos si se mide un lote menor que el guardado."""
    saved = torch.load(path)
    if saved["batch_size"] % size:
        raise ValueError("El lote medido debe dividir el lote guardado")
    batches = []
    for batch in saved["batches"]:
        for part in range(saved["batch_size"] // size):
            rows = slice(part * size, (part + 1) * size)
            batches.append(
                dict(
                    inputs={k: v[rows].numpy() for k, v in batch["inputs"].items()},
                    presence=batch["presence"][rows].numpy(),
                    target=batch["target"][rows].numpy(),
                )
            )
    return saved, batches


def measure(args):
    apply_precision(args.precision)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita CUDA")
    size = args.batch_size
    saved, loaded = stored_batches(args.batches, size)
    batches = [resident(batch, device) for batch in loaded]
    dimensions = {name: value.shape[-1] for name, value in loaded[0]["inputs"].items()}
    cases = [item for item in CASES if not args.cases or item[0] in args.cases]
    results, outputs = {}, {}
    for name, case in cases:
        model = build(case, dimensions, saved["context"], size).to(device)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        model.train()
        train = timed(
            lambda batch, m=model: train_step(m, batch),
            batches,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        peak = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        used = process_mib()
        model.eval()
        with torch.no_grad():
            inference = timed(
                lambda batch, m=model: m(batch["inputs"], batch["presence"]),
                batches,
                warmup=args.warmup,
                repeats=args.repeats,
            )
        with torch.no_grad():
            evaluated = model(batches[0]["inputs"], batches[0]["presence"]).float().cpu()
        model.train()
        trained = model(batches[0]["inputs"], batches[0]["presence"]).detach().float().cpu()
        outputs[name] = dict(eval=evaluated, train=trained)
        forward_syncs = count_syncs(lambda m=model: m(batches[0]["inputs"], batches[0]["presence"]))
        step_syncs = count_syncs(lambda m=model: loop_step(m, loaded[0], device))
        copied = copied_rate(model, loaded, size=size, warmup=args.warmup, device=device)
        results[name] = dict(
            kind=case["kind"],
            architecture=case["architecture"],
            train_compute=summary(train, size),
            inference_compute=summary(inference, size),
            train_with_copies_samples_per_second=copied,
            peak_train_allocated_mib=peak / 2**20,
            peak_train_reserved_mib=reserved / 2**20,
            process_mib=used,
            syncs_per_forward=forward_syncs,
            syncs_per_loop_step_without_optimizer=step_syncs,
        )
        print(name, json.dumps(results[name]), flush=True)
        del model
        torch.cuda.empty_cache()
    if args.outputs:
        torch.save(outputs, args.outputs)
    report = dict(
        code=str(Path(reference_run.__file__).resolve().parents[3]),
        batches_file=str(args.batches),
        loader=saved["loader"],
        stored_batch_size=saved["batch_size"],
        batch_size=size,
        resident_batches=len(batches),
        warmup=args.warmup,
        repeats=args.repeats,
        precision=args.precision,
        float32_matmul_precision=torch.get_float32_matmul_precision(),
        max_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        device=torch.cuda.get_device_name(0),
        python=platform.python_version(),
        matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_tf32=torch.backends.cudnn.allow_tf32,
        mha_fastpath=torch.backends.mha.get_fastpath_enabled(),
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    phase = commands.add_parser("materialize")
    phase.add_argument("view", type=Path)
    phase.add_argument("--output", type=Path, required=True)
    phase.add_argument("--batch-size", type=int, default=512)
    phase.add_argument("--batches", type=int, default=32)
    phase.add_argument("--seed", type=int, default=42)
    phase = commands.add_parser("measure")
    phase.add_argument("batches", type=Path)
    phase.add_argument("--output", type=Path, required=True)
    phase.add_argument("--batch-size", type=int, default=256)
    phase.add_argument("--warmup", type=int, default=3)
    phase.add_argument("--repeats", type=int, default=3)
    phase.add_argument("--cases", nargs="*")
    phase.add_argument("--outputs", type=Path, help="guardar salidas del primer lote")
    phase.add_argument("--precision", choices=("strict", "default"), default="strict")
    args = parser.parse_args()
    (materialize if args.command == "materialize" else measure)(args)


if __name__ == "__main__":
    main()
