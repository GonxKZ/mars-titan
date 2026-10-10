"""Precisión y núcleos de las referencias neuronales sobre lotes reales, sin aprender.

Para cada caso compara FP32 estricto (la política de la campaña) frente a una copia FP64
de los mismos pesos. FP32 con TF32 en cuDNN (el valor por defecto de PyTorch), TF32 y
autocast BF16 se miden solo para documentar por qué se descartan: salida, pérdida pinball,
gradiente de cada lote y gradiente acumulado en varios lotes sin optimizador. También
comprueba cuDNN en las recurrentes, el backend de SDPA y el pico de memoria por lote.

    PYTHONPATH=src python benchmarks/reference_numerics.py lotes.pt --output numerica.json
"""

import argparse
import contextlib
import copy
import json
import statistics
import sys
import time
import warnings
from pathlib import Path

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

sys.path.insert(0, str(Path(__file__).parent))
from reference_kernels import (  # noqa: E402
    CASES,
    apply_precision,
    build,
    resident,
    stored_batches,
)

from mars_titan.models.quantile_head import pinball_loss  # noqa: E402

MODES = {
    "fp32_strict": dict(matmul=False, cudnn=False, autocast=False),
    "fp32_default": dict(matmul=False, cudnn=True, autocast=False),
    "tf32": dict(matmul=True, cudnn=True, autocast=False),
    "bf16_autocast": dict(matmul=False, cudnn=True, autocast=True),
}


@contextlib.contextmanager
def precision(mode, kind):
    settings = MODES[mode]
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = settings["matmul"]
    torch.backends.cudnn.allow_tf32 = settings["cudnn"]
    guard = torch.is_autocast_enabled
    try:
        if settings["autocast"] and kind == "transformer":
            # El contrato del Transformer rechaza autocast. Solo para medir su efecto.
            torch.is_autocast_enabled = lambda *args: False
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=settings["autocast"]):
            yield
    finally:
        torch.is_autocast_enabled = guard
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def as_double(batch):
    return dict(
        inputs={k: v.double() for k, v in batch["inputs"].items()},
        presence=batch["presence"],
        target=batch["target"].double(),
    )


def evaluate(model, batch, mode=None, kind=None):
    with precision(mode, kind) if mode else contextlib.nullcontext():
        emitted = model(batch["inputs"], batch["presence"])
    loss = pinball_loss(emitted.to(batch["target"].dtype), batch["target"])
    gradients = torch.autograd.grad(loss, list(model.parameters()), allow_unused=True)
    gradients = [
        torch.zeros_like(p) if g is None else g
        for p, g in zip(model.parameters(), gradients, strict=True)
    ]
    return emitted.detach(), loss.detach(), torch.cat([g.reshape(-1) for g in gradients])


def relative(value, reference):
    return ((value.double() - reference).norm() / reference.norm().clamp_min(1e-300)).item()


def step_time(model, batches, mode, kind, repeats=3):
    def step(batch):
        with precision(mode, kind):
            emitted = model(batch["inputs"], batch["presence"])
        pinball_loss(emitted.float(), batch["target"]).backward()
        model.zero_grad(set_to_none=True)

    for batch in batches[:2]:
        step(batch)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for batch in batches[2:]:
            step(batch)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) / len(batches[2:]) * 1e3)
    return statistics.median(values)


def precision_study(model, batches, kind):
    # Sin dropout en la copia numérica: las máscaras de FP32 y FP64 no coinciden en CUDA.
    numeric = copy.deepcopy(model)
    for module in numeric.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0.0
    reference = copy.deepcopy(numeric).double()
    exact = [evaluate(reference, as_double(batch)) for batch in batches]
    accumulated = torch.stack([gradient for _, _, gradient in exact]).sum(0)
    result = {}
    for mode in MODES:
        observed = [evaluate(numeric, batch, mode, kind) for batch in batches]
        total = torch.stack([gradient for _, _, gradient in observed]).sum(0)
        result[mode] = dict(
            output_max_abs_error=max(
                (o.double() - e).abs().max().item()
                for (o, _, _), (e, _, _) in zip(observed, exact, strict=True)
            ),
            loss_max_relative_error=max(
                abs(o.item() - e.item()) / abs(e.item())
                for (_, o, _), (_, e, _) in zip(observed, exact, strict=True)
            ),
            gradient_max_relative_error=max(
                relative(o, e) for (_, _, o), (_, _, e) in zip(observed, exact, strict=True)
            ),
            accumulated_gradient_relative_error=relative(total, accumulated),
            train_step_ms=step_time(model, batches, mode, kind),
        )
    return result


def cudnn_report(model, batch):
    """Comprobar `_cudnn_rnn` y que los pesos recurrentes formen un único bloque."""
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with torch.profiler.profile(activities=activities) as prof:
            emitted = model(batch["inputs"], batch["presence"])
            pinball_loss(emitted, batch["target"]).backward()
            torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    names = {event.key for event in prof.key_averages()}
    return dict(
        cudnn_enabled=torch.backends.cudnn.enabled,
        cudnn_rnn_forward="aten::_cudnn_rnn" in names,
        cudnn_rnn_backward="aten::_cudnn_rnn_backward" in names,
        rnn_kernels=sorted(name[:80] for name in names if "rnn" in name.lower())[:8],
        noncontiguous_weight_warnings=sum(
            "contiguous chunk" in str(item.message) for item in caught
        ),
    )


def sdpa_report(model, batches):
    """Kernel de atención elegido y coste del paso con cada backend admitido."""
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        emitted = model(batches[0]["inputs"], batches[0]["presence"])
        pinball_loss(emitted, batches[0]["target"]).backward()
        torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    kernels = sorted(
        {
            event.key[:90]
            for event in prof.key_averages()
            if any(word in event.key.lower() for word in ("fmha", "flash", "attention", "softmax"))
        }
    )
    timings = {}
    for backend in (SDPBackend.EFFICIENT_ATTENTION, SDPBackend.FLASH_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel([backend]):
                timings[backend.name] = step_time(model, batches, "fp32_strict", "transformer")
        except RuntimeError as error:
            timings[backend.name] = f"no disponible: {str(error).splitlines()[0][:120]}"
    return dict(kernels_fp32=kernels, train_step_ms_by_backend_fp32=timings)


def peak_memory(model, batch, sizes):
    peaks = {}
    for size in sizes:
        repeat = -(-size // len(batch["target"]))
        tiled = dict(
            inputs={
                k: v.repeat(repeat, *([1] * (v.ndim - 1)))[:size]
                for k, v in batch["inputs"].items()
            },
            presence=batch["presence"].repeat(repeat, 1)[:size],
            target=batch["target"].repeat(repeat)[:size],
        )
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        try:
            emitted = model(tiled["inputs"], tiled["presence"])
            pinball_loss(emitted, tiled["target"]).backward()
            torch.cuda.synchronize()
            peaks[size] = (torch.cuda.max_memory_allocated() - base) / 2**20
        except (torch.OutOfMemoryError, ValueError) as error:
            peaks[size] = f"{type(error).__name__}: {str(error).splitlines()[0][:100]}"
        model.zero_grad(set_to_none=True)
        del tiled
    return peaks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("batches_file", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--cases", nargs="*")
    parser.add_argument("--memory-sizes", type=int, nargs="*", default=[256, 512])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita CUDA")
    device = torch.device("cuda:0")
    apply_precision("strict")
    saved, loaded = stored_batches(args.batches_file, args.batch_size)
    loaded = loaded[: args.batches]
    batches = [resident(batch, device) for batch in loaded]
    dimensions = {name: value.shape[-1] for name, value in loaded[0]["inputs"].items()}
    results = {}
    for name, case in CASES:
        if args.cases and name not in args.cases:
            continue
        size = max(args.batch_size, max(args.memory_sizes))
        model = build(case, dimensions, saved["context"], size).to(device).train()
        entry = dict(kind=case["kind"], architecture=case["architecture"])
        entry["precision"] = precision_study(model, batches, case["kind"])
        if case["kind"] in ("rnn", "lstm", "gru"):
            entry["cudnn"] = cudnn_report(model, batches[0])
        if case["kind"] == "transformer":
            entry["sdpa"] = sdpa_report(model, batches)
        entry["peak_train_mib_by_batch"] = peak_memory(model, batches[0], args.memory_sizes)
        results[name] = entry
        print(name, json.dumps(entry), flush=True)
        del model
        torch.cuda.empty_cache()
    report = dict(
        batch_size=args.batch_size,
        batches=len(batches),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        device=torch.cuda.get_device_name(0),
        total_memory_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
        results=results,
    )
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
