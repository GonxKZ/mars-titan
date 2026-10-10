"""torch.compile y CUDA Graphs en forward, pinball y backward de las referencias, sin optimizador.

Usa los lotes guardados por `reference_kernels.py materialize` con el lote declarado de 256 y
FP32 estricto. Para cada caso y modo mide el paso, el tiempo de la primera llamada y la
diferencia máxima de la salida del primer lote frente al modo eager.

    PYTHONPATH=src python benchmarks/reference_graphs.py lotes.pt gru-00,transformer-00 \\
        eager,graphs,default,max-autotune-no-cudagraphs resultado.json
"""

import json
import sys
import time
from pathlib import Path

import torch
from reference_kernels import CASES, apply_precision, build, resident, stored_batches

from mars_titan.models.quantile_head import pinball_loss

SIZE = 256


def timed(step, batches, warmup=3, repeats=3):
    for batch in batches[:warmup]:
        step(batch)
    torch.cuda.synchronize()
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        for batch in batches[warmup:]:
            step(batch)
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) / len(batches[warmup:]) * 1e3)
    return sorted(values)[len(values) // 2]


def graphed(model, batches):
    """Capturar forward y backward con entradas estáticas y devolver el paso y la salida."""
    first = batches[0]
    static = {k: v.clone() for k, v in first["inputs"].items()}
    presence, target = first["presence"].clone(), first["target"].clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pinball_loss(model(static, presence), target).backward()
            model.zero_grad(set_to_none=False)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = model(static, presence)
        pinball_loss(output, target).backward()

    def step(batch):
        for k, v in batch["inputs"].items():
            static[k].copy_(v)
        presence.copy_(batch["presence"])
        target.copy_(batch["target"])
        graph.replay()

    return step, output


def measure(model, mode, batches, reference):
    record = {}
    if mode == "graphs":
        step, output = graphed(model, batches)
        record["step_ms"] = timed(step, batches)
        step(batches[0])
        torch.cuda.synchronize()
    else:
        forward = model if mode == "eager" else torch.compile(model, mode=mode)

        def step(batch):
            pinball_loss(forward(batch["inputs"], batch["presence"]), batch["target"]).backward()
            model.zero_grad(set_to_none=True)

        start = time.perf_counter()
        step(batches[0])
        torch.cuda.synchronize()
        record["first_call_s"] = time.perf_counter() - start
        record["step_ms"] = timed(step, batches)
        output = forward(batches[0]["inputs"], batches[0]["presence"])
    if reference is not None:
        record["max_abs_vs_eager"] = (output.detach() - reference).abs().max().item()
    return record, output.detach()


def main():
    source, names, modes, target = sys.argv[1:5]
    device = torch.device("cuda:0")
    apply_precision("strict")
    saved, loaded = stored_batches(Path(source), SIZE)
    batches = [resident(batch, device) for batch in loaded[:12]]
    dimensions = {k: v.shape[-1] for k, v in loaded[0]["inputs"].items()}
    cases, results = dict(CASES), {}
    for name in names.split(","):
        reference = None
        for mode in modes.split(","):
            torch.compiler.reset()
            model = build(cases[name], dimensions, saved["context"], SIZE).to(device).train()
            try:
                record, output = measure(model, mode, batches, reference)
                if mode == "eager":
                    reference = output
            except Exception as error:  # noqa: BLE001
                record = dict(error=f"{type(error).__name__}: {str(error).splitlines()[0][:200]}")
            results[f"{name}:{mode}"] = dict(case=name, mode=mode, **record)
            print(json.dumps(results[f"{name}:{mode}"]), flush=True)
            torch.cuda.empty_cache()
    Path(target).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
