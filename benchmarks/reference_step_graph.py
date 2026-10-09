"""Paso de `reference_run` con y sin CUDA Graphs sobre lotes reales, sin optimizador.

Usa los lotes guardados por `reference_kernels.py materialize` con el lote declarado de 256
y FP32 estricto. Para cada caso recorre los lotes con la ruta eager y con
`ReferenceStepGraph`, igual que el bucle de `reference_run` hasta el optimizador (copias,
forward, pérdida, comprobaciones, backward, recorte y estadísticas). Comprueba que las
predicciones y los gradientes de cada lote son idénticos bit a bit y mide el tiempo por paso
alternando ambas rutas.

    PYTHONPATH=src python benchmarks/reference_step_graph.py lotes.pt resultado.json
"""

import json
import math
import sys
import time

import torch
from reference_kernels import CASES, apply_precision, build, stored_batches

from mars_titan.training import reference_run
from mars_titan.training.reference_step_graph import ReferenceStepGraph

SIZE, WARMUP, REPEATS = 256, 3, 5
CASE_NAMES = ("rnn-10", "lstm-10", "gru-10", "dlinear-10", "transformer-10", "transformer-11")


def walk(model, batches, graph, *, record=True):
    """Recorrer los lotes y devolver predicciones y gradientes de cada paso."""
    device = torch.device("cuda:0")
    loss = reference_run._training_loss(dict(loss="pinball", head="quantile_head_v1"), True)
    statistics = reference_run._statistics()
    records = [] if record else None
    for batch in batches:
        target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
        if graph is not None and graph.admits(batch):
            prediction = graph.step(batch, target)
        else:
            model.zero_grad(set_to_none=True)
            emitted = reference_run._forward(model, batch, device)
            prediction, value = loss(emitted, target, None)
            if not torch.isfinite(value).item():
                raise ValueError("La pérdida no es finita")
            value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
        reference_run._update(statistics, prediction, torch.from_numpy(batch["target"]).to(device))
        if records is not None:
            records.append((prediction.clone(), [p.grad.clone() for p in model.parameters()]))
    return records


def timed(model, batches, graph):
    walk(model, batches[:WARMUP], graph, record=False)
    torch.cuda.synchronize()
    start = time.perf_counter()
    walk(model, batches[WARMUP:], graph, record=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / len(batches[WARMUP:]) * 1e3


def main():
    apply_precision("strict")
    torch.use_deterministic_algorithms(True)
    saved, batches = stored_batches(sys.argv[1], SIZE)
    dimensions = {name: value.shape[-1] for name, value in batches[0]["inputs"].items()}
    loss = reference_run._training_loss(dict(loss="pinball", head="quantile_head_v1"), True)
    results = {}
    for name, case in CASES:
        if name not in CASE_NAMES:
            continue
        models = {}
        for mode in ("eager", "graph"):
            model = build(case, dimensions, saved["context"], SIZE).to("cuda:0").train()
            models[mode] = (
                model,
                ReferenceStepGraph(model, loss, SIZE) if mode == "graph" else None,
            )
        outputs = {}
        for mode, (model, graph) in models.items():
            torch.cuda.manual_seed(7)
            outputs[mode] = walk(model, batches[:8], graph)
        identical = all(
            torch.equal(a[0], b[0])
            and all(torch.equal(x, y) for x, y in zip(a[1], b[1], strict=True))
            for a, b in zip(outputs["eager"], outputs["graph"], strict=True)
        )
        times = {"eager": [], "graph": []}
        for _ in range(REPEATS):
            for mode, (model, graph) in models.items():
                times[mode].append(timed(model, batches, graph))
        results[name] = dict(
            identical_predictions_and_gradients=identical,
            step_ms={mode: sorted(values) for mode, values in times.items()},
            median_ms={mode: sorted(values)[len(values) // 2] for mode, values in times.items()},
            replays=models["graph"][1].replays,
        )
        results[name]["speedup"] = (
            results[name]["median_ms"]["eager"] / results[name]["median_ms"]["graph"]
        )
        print(name, json.dumps(results[name]), flush=True)
        del models, outputs
        torch.cuda.empty_cache()
    report = dict(
        batch_size=SIZE,
        batches=len(batches),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),
        gpu=torch.cuda.get_device_name(0),
        results=results,
    )
    with open(sys.argv[2], "w") as handle:
        json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
