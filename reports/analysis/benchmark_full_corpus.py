"""Medir pasos reales desde disco, sin confundir la sonda con un modelo seleccionado."""

import argparse
import gc
import json
import math
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from mars_titan.budget_training import seed_run
from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.models.baselines.multimodal import MultimodalReference
from mars_titan.training.corpus_inputs import CorpusDataset
from mars_titan.training.reference_run import _inputs, _statistics, _update, scientific_identity


def step(model, optimizer, batch, device, statistics):
    optimizer.zero_grad(set_to_none=True)
    target = torch.from_numpy(batch["target"]).to(device, dtype=torch.float32)
    predicted = model(_inputs(batch, device))
    loss = torch.nn.functional.mse_loss(predicted, target)
    if not torch.isfinite(loss).item():
        raise ValueError("La pérdida de la sonda no es finita")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), math.inf, error_if_nonfinite=True)
    optimizer.step()
    _update(statistics, predicted, torch.from_numpy(batch["target"]).to(device))


def benchmark(manifest, output):
    started = time.perf_counter()
    device = require_cuda()
    seed_run(42)
    dataset = CorpusDataset(manifest)
    if dataset.manifest["scope"] != "full_corpus" or not dataset.manifest["cohort_complete"]:
        raise ValueError("La medición necesita el manifiesto completo confirmado")
    safe_destination(output)
    for root in (*dataset.roots.values(), manifest.parent):
        outside_source(root, output)
        outside_source(output, root)
    output.mkdir(exist_ok=False, parents=True)
    report = dict(
        status="running",
        purpose="resource_profiling_not_model_selection",
        source_sha256=sha256(manifest),
        counts=dataset.manifest["counts"],
        assets=len(dataset.assets),
        scope="full_corpus",
        code_sha256=sha256(Path(__file__)),
        scientific_identity=scientific_identity(),
        initialization_seconds=time.perf_counter() - started,
        hardware_before=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.free,utilization.gpu,power.draw",
                "--format=csv",
            ],
            text=True,
        ),
        concurrent_workloads=subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv",
            ],
            text=True,
        ),
        cases=[],
        final_test_opened=False,
        limits=[
            "La sonda consume ventanas reales de train, pero no recorre toda una época.",
            "La caché del sistema operativo y la carga de otras aplicaciones no se controlan.",
            "Los intervalos de pasos incluyen lectura, transferencias, gradientes y estadísticas.",
            "No incluyen evaluación completa ni escritura de checkpoints.",
            "No se infiere consumo energético a partir de la muestra instantánea de potencia.",
        ],
    )
    atomic_json(output / "progress.json", report)
    try:
        for kind in ("rnn", "lstm", "gru", "dlinear"):
            for batch_size in (256, 512):
                for repeat in range(3):
                    seed_run(42)
                    reader = dataset.batches(
                        partition="train", batch_size=batch_size, epoch=0, seed=42
                    )
                    first = next(reader)
                    dimensions = {
                        name: values.shape[-1] for name, values in first["inputs"].items()
                    }
                    model = MultimodalReference(
                        kind,
                        dimensions,
                        context=dataset.context,
                        hidden_size=128,
                        layers=2,
                        dropout=0.2,
                    ).to(device)
                    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                    statistics = _statistics()

                    for _ in range(5):
                        step(model, optimizer, first, device, statistics)
                    torch.cuda.synchronize(0)
                    torch.cuda.reset_peak_memory_stats(0)
                    times, read_times, rows, transferred = [], [], 0, 0
                    sample_ids, asset_ids = set(), set()
                    for _ in range(24):
                        before = time.perf_counter()
                        batch = next(reader)
                        read_times.append(time.perf_counter() - before)
                        step(model, optimizer, batch, device, statistics)
                        torch.cuda.synchronize(0)
                        times.append(time.perf_counter() - before)
                        rows += len(batch["target"])
                        sample_ids.update(batch["sample_ids"])
                        asset_ids.update(
                            "/".join(key.split("/")[:2]) for key in batch["sample_ids"]
                        )
                        transferred += (
                            sum(a.nbytes for a in batch["inputs"].values())
                            + len(batch["target"]) * 12
                        )
                    record = dict(
                        kind=kind,
                        batch_size=batch_size,
                        repeat=repeat,
                        steps=len(times),
                        rows=rows,
                        unique_sample_count=len(sample_ids),
                        unique_asset_count=len(asset_ids),
                        seconds=sum(times),
                        read_seconds=sum(read_times),
                        transfer_update_statistics_seconds=sum(times) - sum(read_times),
                        samples_per_second=rows / sum(times),
                        step_p50_ms=float(np.quantile(times, 0.5) * 1000),
                        step_p95_ms=float(np.quantile(times, 0.95) * 1000),
                        step_seconds=times,
                        read_step_seconds=read_times,
                        warmup_steps=5,
                        dimensions=dimensions,
                        architecture=dict(hidden_size=128, layers=2, dropout=0.2),
                        peak_vram_allocated_bytes=torch.cuda.max_memory_allocated(0),
                        peak_vram_reserved_bytes=torch.cuda.max_memory_reserved(0),
                        process_lifetime_peak_rss_bytes=resource.getrusage(
                            resource.RUSAGE_SELF
                        ).ru_maxrss
                        * 1024,
                        logical_host_to_device_bytes=transferred,
                        physical_pcie_bytes_measured=False,
                    )
                    report["cases"].append(record)
                    atomic_json(output / "progress.json", report)
                    print(json.dumps(record), flush=True)
                    reader.close()
                    del reader, model, optimizer, first, batch, statistics
                    gc.collect()
                    torch.cuda.empty_cache()

    except BaseException as error:
        report.update(status="failed", failure=dict(type=type(error).__name__, message=str(error)))
        atomic_json(output / "progress.json", report)
        raise
    report["status"] = "completed"
    report["total_seconds"] = time.perf_counter() - started
    atomic_json(output / "profile.json", report)
    atomic_json(output / "progress.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.manifest, args.output)
