"""Mediciones locales de coste, separadas de los experimentos predictivos."""

import json
import resource
import statistics
import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from mars_titan.data.embeddings import require_cuda
from mars_titan.data.storage import atomic_json
from mars_titan.data.streaming import iter_windows

MODALITIES = ("prices", "news", "charts", "fundamentals", "macro")


def _step(model, optimizer, batch, device):
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    optimizer.zero_grad(set_to_none=True)
    output = model(batch)
    # Synthetic zero targets measure backward cost, never predictive accuracy.
    loss = output.square().mean()
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite cost-probe loss")
    loss.backward()
    optimizer.step()


class CostProbe(nn.Module):
    """MLP o GRU pequeña para medir entrenamiento con todas las entradas."""

    def __init__(self, kind: str, dimensions: dict[str, int], context: int = 64):
        super().__init__()
        if kind not in {"mlp", "gru"} or set(dimensions) != set(MODALITIES):
            raise ValueError("Invalid cost probe or modalities")
        self.kind = kind
        self.price_encoder = (
            nn.GRU(dimensions["prices"], 32, batch_first=True)
            if kind == "gru"
            else nn.Sequential(
                nn.Flatten(1), nn.Linear(context * dimensions["prices"], 32), nn.SiLU()
            )
        )
        self.encoders = nn.ModuleDict(
            {
                name: nn.Sequential(nn.Linear(dimensions[name], 32), nn.SiLU())
                for name in MODALITIES
                if name != "prices"
            }
        )
        self.head = nn.Sequential(nn.Linear(160, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        if set(inputs) != set(MODALITIES):
            raise ValueError("All four modalities and macro are required")
        if self.kind == "gru":
            _, hidden = self.price_encoder(inputs["prices"])
            price = hidden[-1]
        else:
            price = self.price_encoder(inputs["prices"])
        representations = [price] + [self.encoders[name](inputs[name]) for name in self.encoders]
        return self.head(torch.cat(representations, dim=-1)).squeeze(-1)


class WindowWorkload(IterableDataset):
    def __init__(self, paths: list[Path], prepared: Path, cutoff: str = "2023-12-31"):
        self.paths, self.prepared, self.cutoff = paths, prepared, cutoff

    def __iter__(self):
        worker = get_worker_info()
        for record in iter_windows(
            self.paths,
            self.prepared,
            worker=worker.id if worker else 0,
            workers=worker.num_workers if worker else 1,
            decision_cutoff=self.cutoff,
        ):
            yield record["inputs"]


def project_cost(
    *, samples: int, seconds_per_sample: float, epochs: int, folds: int, seeds: int, models: int
) -> dict:
    if min(samples, seconds_per_sample, epochs, folds, seeds, models) <= 0:
        raise ValueError("Projection inputs must be positive")
    return {
        "training_seconds": samples * seconds_per_sample * epochs * folds * seeds * models,
        "architecture_scope": "measured_cost_probes_only",
        "excluded_costs": [
            "embedding_extraction",
            "validation",
            "hyperparameter_search",
            "checkpoint_io",
            "future_memory_architecture",
        ],
    }


def profile_case(
    paths: list[Path],
    prepared: Path,
    *,
    kind: str,
    batch_size: int,
    workers: int,
    steps: int = 50,
    repeat: int = 0,
) -> dict:
    if not paths or min(batch_size, steps) < 1 or workers not in {0, 2, 4}:
        raise ValueError("Invalid profiling workload")
    device = require_cuda()
    torch.manual_seed(42 + repeat)
    torch.set_num_threads(4)
    loader = DataLoader(
        WindowWorkload(paths, prepared),
        batch_size=batch_size,
        num_workers=workers,
        pin_memory=True,
        **({"prefetch_factor": 2, "multiprocessing_context": "spawn"} if workers else {}),
    )
    startup = time.perf_counter()
    iterator = iter(loader)
    first = next(iterator, None)
    if first is None:
        raise ValueError("No complete multimodal samples before profiling cutoff")
    dimensions = {key: value.shape[-1] for key, value in first.items()}
    model = CostProbe(kind, dimensions).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    for _ in range(5):
        _step(model, optimizer, first, device)
    torch.cuda.synchronize()
    startup_seconds = time.perf_counter() - startup
    torch.cuda.reset_peak_memory_stats()
    latencies, samples = [], 0
    for _ in range(steps):
        started = time.perf_counter()
        batch = next(iterator, None)
        if batch is None:
            break
        _step(model, optimizer, batch, device)
        torch.cuda.synchronize()
        latencies.append(time.perf_counter() - started)
        samples += batch["prices"].shape[0]
    if not samples:
        raise ValueError("Too few samples for a measured step after warmup")
    elapsed = sum(latencies)
    ordered = sorted(latencies)
    result = {
        "kind": kind,
        "batch_size": batch_size,
        "workers": workers,
        "repeat": repeat,
        "steps": len(latencies),
        "samples": samples,
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "seconds_per_sample": elapsed / samples,
        "step_p50_ms": statistics.median(latencies) * 1000,
        "step_p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1000,
        "startup_and_warmup_seconds": startup_seconds,
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "parameters": sum(p.numel() for p in model.parameters()),
        "dimensions": dimensions,
        "context": 64,
        "precision": "float32",
        "cutoff": "2023-12-31",
        "targets": "synthetic_zero_cost_only",
        "cache_state": "operating_system_cache_uncontrolled",
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    # Finish file readers naturally before worker processes shut down.
    for _ in iterator:
        pass
    del iterator, loader, optimizer, model, first, batch
    torch.cuda.empty_cache()
    return result


def profile_grid(
    paths: list[Path], prepared: Path, report_path: Path, *, steps: int = 50, repeats: int = 3
) -> dict:
    report = {"schema_version": 1, "purpose": "engineering_cost_only", "cases": []}
    for kind in ("mlp", "gru"):
        for workers in (0, 2, 4):
            for batch_size in (16, 32, 64):
                for repeat in range(repeats):
                    case = profile_case(
                        paths,
                        prepared,
                        kind=kind,
                        workers=workers,
                        batch_size=batch_size,
                        steps=steps,
                        repeat=repeat,
                    )
                    report["cases"].append(case)
                    atomic_json(report_path, report)
                    print(
                        json.dumps(
                            {
                                k: case[k]
                                for k in (
                                    "kind",
                                    "workers",
                                    "batch_size",
                                    "repeat",
                                    "samples_per_second",
                                )
                            }
                        ),
                        flush=True,
                    )
    return report
