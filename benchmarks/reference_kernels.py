"""Caudal de las referencias neuronales de la campaña A sobre lotes reales, sin aprender.

Mide por caso el paso forward, pinball y backward sin optimizador (los gradientes se
liberan sin tocar pesos) y la inferencia sin gradiente, primero con los lotes ya
residentes en la GPU (cómputo) y después leyendo cada lote del corpus como el ajuste
(con carga). Cuenta además las sincronizaciones de un paso del bucle de
`reference_run` hasta el optimizador. Sirve para contrastar dos versiones del código:
basta con cambiar `PYTHONPATH` y alternar las ejecuciones.

    PYTHONPATH=src python benchmarks/reference_kernels.py VIEW --output resultado.json
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


def loading_rate(model, dataset, *, size, seed, warmup, batches, device):
    """Como `campaign_throughput._rate`: lectura, copia, forward, pinball y backward."""
    model.train()
    rows, start = 0, None
    source = dataset.batches(partition="train", batch_size=size, epoch=0, seed=seed)
    for index, batch in enumerate(source):
        if index == warmup:
            torch.cuda.synchronize()
            start = time.perf_counter()
        if index == warmup + batches:
            break
        emitted = reference_run._forward(model, batch, device)
        target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
        pinball_loss(emitted, target).backward()
        model.zero_grad(set_to_none=True)
        rows += len(batch["target"]) if index >= warmup else 0
    torch.cuda.synchronize()
    return rows / (time.perf_counter() - start)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("view", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--resident", type=int, default=23)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--loading-batches", type=int, default=40)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--outputs", type=Path, help="guardar salidas del primer lote")
    parser.add_argument("--precision", choices=("strict", "default"), default="strict")
    parser.add_argument("--save-batches", type=Path, help="guardar los lotes residentes")
    args = parser.parse_args()
    apply_precision(args.precision)
    device = torch.device("cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita CUDA")
    dataset = reference_run.configured_corpus(args.view, input_policy=INPUT_POLICY)
    size = args.batch_size
    start = time.perf_counter()
    loaded = []
    for batch in dataset.batches(partition="train", batch_size=size, epoch=0, seed=42):
        loaded.append(batch)
        if len(loaded) == args.resident:
            break
    load_ms = (time.perf_counter() - start) / len(loaded) * 1e3
    batches = [resident(batch, device) for batch in loaded]
    dimensions = {name: value.shape[-1] for name, value in loaded[0]["inputs"].items()}
    if args.save_batches:
        # Lotes en CPU para los procesos concurrentes, que así no leen el corpus.
        torch.save(
            dict(
                context=dataset.context,
                batches=[
                    dict(
                        inputs={k: v.cpu() for k, v in batch["inputs"].items()},
                        presence=batch["presence"].cpu(),
                        target=batch["target"].cpu(),
                    )
                    for batch in batches
                ],
            ),
            args.save_batches,
        )
    cases = [item for item in CASES if not args.cases or item[0] in args.cases]
    results, outputs = {}, {}
    for name, case in cases:
        model = build(case, dimensions, dataset.context, size).to(device)
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
        loading = loading_rate(
            model,
            dataset,
            size=size,
            seed=case["seed"],
            warmup=args.warmup,
            batches=args.loading_batches,
            device=device,
        )
        results[name] = dict(
            kind=case["kind"],
            architecture=case["architecture"],
            train_compute=summary(train, size),
            inference_compute=summary(inference, size),
            train_with_loading_samples_per_second=loading,
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
        batch_size=size,
        resident_batches=len(batches),
        warmup=args.warmup,
        repeats=args.repeats,
        loading_batches=args.loading_batches,
        serial_load_ms_per_batch=load_ms,
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


if __name__ == "__main__":
    main()
