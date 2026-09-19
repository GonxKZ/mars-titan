"""Entrenamientos supervisados breves para estimar recursos, con recuperación."""

import json
import os
import random
import subprocess
import tempfile
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from mars_titan.data.budget_targets import residual_targets
from mars_titan.data.embeddings import require_cuda
from mars_titan.data.prices import read_prices
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.streaming import iter_windows
from mars_titan.data.temporal import MarketClock
from mars_titan.profiling import CostProbe


def seed_run(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)


def target_partition(prediction_at, target_available_at):
    if prediction_at.year <= 2022 and target_available_at.year <= 2022:
        return "train"
    if prediction_at.year == 2023 and target_available_at.year == 2023:
        return "validation"
    return None


def train_step(model, optimizer, batch, device):
    inputs, target = batch
    inputs = {key: value.to(device) for key, value in inputs.items()}
    target = target.to(device)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = torch.nn.functional.mse_loss(prediction, target)
    if not torch.isfinite(loss):
        raise ValueError("Nonfinite supervised cost loss")
    loss.backward()
    optimizer.step()
    return float(loss.detach())


def save_checkpoint(path, model, optimizer, *, next_epoch, config, hashes):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = np.random.get_state()
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "next_epoch": next_epoch,
        "config": config,
        "input_hashes": hashes,
        "rng_python": random.getstate(),
        "rng_torch": torch.get_rng_state(),
        "rng_cuda": torch.cuda.get_rng_state_all(),
        "rng_numpy": [state[0], torch.from_numpy(state[1].astype(np.int64)), *state[2:]],
        "persistent_memory": None,
        "pending_labels": [],
        "confirmed_cursor": "epoch_boundary_replay_next_epoch_from_start",
    }
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_checkpoint(path, model, optimizer, *, config, hashes):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["config"] != config or state["input_hashes"] != hashes:
        raise ValueError("Checkpoint inputs or configuration changed")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["rng_python"])
    numpy_state = state["rng_numpy"]
    np.random.set_state(
        (numpy_state[0], numpy_state[1].numpy().astype(np.uint32), *numpy_state[2:])
    )
    torch.set_rng_state(state["rng_torch"])
    torch.cuda.set_rng_state_all(state["rng_cuda"])
    return state["next_epoch"]


def process_memory():
    """RSS/PSS del árbol vivo mediante procfs, sin sumar máximos incompatibles."""
    pending, seen = [os.getpid()], set()
    rss = pss = 0
    while pending:
        pid = pending.pop()
        if pid in seen:
            continue
        seen.add(pid)
        try:
            pending.extend(
                int(p) for p in Path(f"/proc/{pid}/task/{pid}/children").read_text().split()
            )
            for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
                if line.startswith("Rss:"):
                    rss += int(line.split()[1])
                elif line.startswith("Pss:"):
                    pss += int(line.split()[1])
        except FileNotFoundError:
            continue  # A child can finish between enumerating and reading it.
    return {"rss_mib": rss / 1024, "pss_mib": pss / 1024, "processes": len(seen)}


def run_epoch(model, optimizer, loader, device):
    model.train(optimizer is not None)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    latencies, losses, samples = [], 0.0, 0
    peak = process_memory()
    stopped = threading.Event()

    def observe():
        while not stopped.wait(0.25):
            reading = process_memory()
            for key, value in reading.items():
                peak[key] = max(peak[key], value)

    monitor = threading.Thread(target=observe, daemon=True)
    monitor.start()
    started = step_started = time.perf_counter()
    try:
        for batch in loader:
            if optimizer is not None:
                loss = train_step(model, optimizer, batch, device)
            else:
                with torch.inference_mode():
                    inputs, target = batch
                    prediction = model({key: value.to(device) for key, value in inputs.items()})
                    loss = float(torch.nn.functional.mse_loss(prediction, target.to(device)))
                    if not np.isfinite(loss):
                        raise ValueError("Nonfinite validation cost loss")
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - step_started)
            count = len(batch[1])
            samples += count
            losses += loss * count
            step_started = time.perf_counter()
    finally:
        elapsed = time.perf_counter() - started
        stopped.set()
        monitor.join()
    if not samples:
        raise ValueError("No usable supervised samples")
    return {
        "samples": samples,
        "steps": len(latencies),
        "elapsed_seconds": elapsed,
        "samples_per_second": samples / elapsed,
        "diagnostic_mse": losses / samples,
        **{f"step_p{q}_ms": float(np.percentile(latencies, q) * 1000) for q in [50, 95, 99]},
        "peak_vram_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_vram_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "sampled_tree_peak": peak,
        "ram_poll_seconds": 0.25,
        "latency_scope": "loader_plus_transfer_plus_synchronized_gpu_step",
    }


def labeled_records(records, targets, partition):
    for row in records:
        label = targets.get(row["cursor"][0], {}).get(row["prediction_at"].isoformat())
        if label is not None and label[1] == partition:
            yield row["inputs"], np.float32(label[0])


class SupervisedWorkload(IterableDataset):
    def __init__(self, paths, prepared, targets, partition):
        self.paths, self.prepared = paths, prepared
        self.targets, self.partition = targets, partition

    def __iter__(self):
        cutoff = "2022-12-31" if self.partition == "train" else "2023-12-31"
        return labeled_records(
            iter_windows(self.paths, self.prepared, decision_cutoff=cutoff),
            self.targets,
            self.partition,
        )


def prepare_targets(paths, prepared, output):
    """Materializa únicamente etiquetas maduras hasta 2023 y audita exclusiones."""
    market_path = Path("dataset/time_series/S&P500_time_series/spy.csv")
    clock = MarketClock("US", "2008-01-01", "2024-01-05")
    market, market_audit = read_prices(market_path, clock)
    targets, audit = {}, {"market_price_audit": market_audit, "assets": {}}
    hashes = {str(market_path): sha256(market_path)}
    output.mkdir(parents=True, exist_ok=True)
    for path in paths:
        symbol = path.parent.name
        price_path = prepared / "US" / symbol / "prices.parquet"
        asset = pq.read_table(price_path, filters=[("session", "<=", "2023-12-31")]).to_pandas()
        labels = residual_targets(asset, market, clock, cutoff="2023-12-31")
        label_path = output / f"{symbol}-targets.parquet"
        labels.to_parquet(label_path, index=False)
        accepted = labels.loc[labels.reason == "accepted"]
        targets[f"US/{symbol}"] = {
            row.prediction_at.isoformat(): (row.target, partition)
            for row in accepted.itertuples()
            if (partition := target_partition(row.prediction_at, row.target_available_at))
        }
        dates = pq.read_table(
            path,
            columns=["prediction_at"],
            filters=[("prediction_at", "<", clock.decision("2024-01-02"))],
        )
        counts = {"train": 0, "validation": 0, "excluded": 0}
        by_time = labels.set_index("prediction_at").reason.to_dict()
        reasons = {}
        for day in dates["prediction_at"].to_pylist():
            label = targets[f"US/{symbol}"].get(day.isoformat())
            counts[label[1] if label else "excluded"] += 1
            if label is None:
                reason = by_time.get(day, "outside_label_calendar")
                if reason == "accepted":
                    reason = "target_crosses_partition_boundary"
                reasons[reason] = reasons.get(reason, 0) + 1
        audit["assets"][symbol] = {
            **counts,
            "excluded_reasons": reasons,
            "calendar_label_reasons": labels.reason.value_counts().to_dict(),
        }
        for artifact in [
            path,
            price_path,
            path.parent / "manifest.json",
            price_path.parent / "manifest.json",
            label_path,
        ]:
            hashes[str(artifact)] = sha256(artifact)
    for source in [
        Path(__file__),
        Path(__file__).parent / "data/budget_targets.py",
        Path(__file__).parent / "data/streaming.py",
        Path(__file__).parent / "profiling.py",
    ]:
        hashes[str(source)] = sha256(source)
    atomic_json(output / "targets-audit.json", audit)
    return targets, audit, hashes


def gpu_sensors():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,temperature.gpu,power.draw,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "available": result.returncode == 0,
        "reading": result.stdout.strip(),
        "error": result.stderr.strip() or None,
    }


def train_budget_grid(
    prepared: Path,
    report_path: Path,
    output: Path,
    *,
    epochs: int = 3,
    panel_sizes: tuple[int, ...] = (4, 11, 22),
    kinds: tuple[str, ...] = ("mlp", "gru"),
):
    """Paneles técnicos completos y reanudación de los casos de cuatro activos."""
    if (
        not 1 <= epochs <= 30
        or not panel_sizes
        or not kinds
        or not set(panel_sizes) <= {4, 11, 22}
        or not set(kinds) <= {"mlp", "gru"}
    ):
        raise ValueError("Invalid bounded budget workload")
    for target in (output, report_path):
        outside_source(Path("dataset"), target)
        outside_source(prepared, target)
    if output.exists():
        raise ValueError("Use a fresh output directory to preserve existing checkpoints")
    started = time.perf_counter()
    started_at = datetime.now(UTC).isoformat()
    device = require_cuda()
    paths = sorted((prepared / "samples/US").glob("*/samples.parquet"))
    if len(paths) != 22:
        raise ValueError("This bounded budget experiment requires the 22 prepared assets")
    targets, audit, hashes = prepare_targets(paths, prepared, output / "targets")
    paths.sort(key=lambda path: (audit["assets"][path.parent.name]["train"] == 0, path.parent.name))
    report = {
        "schema_version": 1,
        "started_at_utc": started_at,
        "purpose": "supervised_engineering_budget_only",
        "target": "next_session_open_close_return_minus_past_OLS_intercept_and_SPY_exposure",
        "history_sessions": 252,
        "minimum_pairs": 126,
        "sector_residualization": False,
        "training_cutoff": "2022-12-31",
        "validation_year": 2023,
        "final_test_opened": False,
        "selection_rule": "alphabetic_train_available_first",
        "seed": 42,
        "epochs": epochs,
        "batch_size": 64,
        "workers": 0,
        "lr": 1e-4,
        "optimizer": {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.01},
        "deterministic_algorithms": True,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "precision": "float32",
        "context": 64,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "input_hashes": hashes,
        "target_audit": audit,
        "preparation_seconds": time.perf_counter() - started,
        "cache_state": "operating_system_cache_uncontrolled",
        "sensor_sampling": "case_boundaries_only_not_energy_integration",
        "excluded_costs": ["embeddings", "source_preparation", "future_MARS_TITAN_architecture"],
        "cases": [],
    }
    atomic_json(report_path, report)
    for size in panel_sizes:
        panel = paths[:size]
        for kind in kinds:
            startup = time.perf_counter()
            seed_run(42)
            loaders = {
                partition: DataLoader(
                    SupervisedWorkload(panel, prepared, targets, partition),
                    batch_size=64,
                    num_workers=0,
                    pin_memory=True,
                )
                for partition in ["train", "validation"]
            }
            first, _ = next(iter(loaders["train"]))
            dims = {key: value.shape[-1] for key, value in first.items()}
            model = CostProbe(kind, dims).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            config = {
                "seed": 42,
                "kind": kind,
                "dimensions": dims,
                "batch_size": 64,
                "symbols": [path.parent.name for path in panel],
                "lr": 1e-4,
                "context": 64,
                "precision": "float32",
                "epochs": epochs,
            }
            case = {
                **config,
                "assets": size,
                "effective_training_assets": sum(
                    audit["assets"][path.parent.name]["train"] > 0 for path in panel
                ),
                "effective_validation_assets": sum(
                    audit["assets"][path.parent.name]["validation"] > 0 for path in panel
                ),
                "sample_counts_by_asset": {
                    path.parent.name: audit["assets"][path.parent.name] for path in panel
                },
                "parameters": sum(p.numel() for p in model.parameters()),
                "startup_seconds": time.perf_counter() - startup,
                "sensors_before": gpu_sensors(),
                "epochs": [],
            }
            case_dir = output / f"{kind}-{size}"
            for epoch in range(epochs):
                training = run_epoch(model, optimizer, loaders["train"], device)
                validation = run_epoch(model, None, loaders["validation"], device)
                checkpoint_started = time.perf_counter()
                checkpoint = case_dir / f"epoch-{epoch + 1}.pt"
                save_checkpoint(
                    checkpoint, model, optimizer, next_epoch=epoch + 1, config=config, hashes=hashes
                )
                measurement = {
                    "epoch": epoch + 1,
                    "train": training,
                    "validation": validation,
                    "checkpoint_seconds": time.perf_counter() - checkpoint_started,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                }
                case["epochs"].append(measurement)
                atomic_json(case_dir / "measurements.json", case)
                print(
                    json.dumps(
                        {
                            "kind": kind,
                            "assets": size,
                            "epoch": epoch + 1,
                            "train_seconds": training["elapsed_seconds"],
                            "validation_seconds": validation["elapsed_seconds"],
                        }
                    ),
                    flush=True,
                )
            case["sensors_after"] = gpu_sensors()
            epoch_costs = [
                entry["train"]["elapsed_seconds"]
                + entry["validation"]["elapsed_seconds"]
                + entry["checkpoint_seconds"]
                for entry in case["epochs"]
            ]
            case["projections_seconds"] = {
                str(n): {
                    "median": case["startup_seconds"] + n * float(np.median(epoch_costs)),
                    "observed_epoch_min": case["startup_seconds"] + n * min(epoch_costs),
                    "observed_epoch_max": case["startup_seconds"] + n * max(epoch_costs),
                }
                for n in [10, 20, 30]
            }
            if size == 4:
                expected = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
                resume_started = time.perf_counter()
                next_epoch = load_checkpoint(
                    case_dir / "epoch-1.pt", model, optimizer, config=config, hashes=hashes
                )
                for _ in range(next_epoch, epochs):
                    run_epoch(model, optimizer, loaders["train"], device)
                    run_epoch(model, None, loaders["validation"], device)
                exact = all(
                    torch.equal(expected[key], value) for key, value in model.state_dict().items()
                )
                if not exact:
                    raise ValueError("Resumed weights differ from uninterrupted training")
                case["resume_check"] = {
                    "exact_weights": exact,
                    "from_epoch": 1,
                    "elapsed_seconds": time.perf_counter() - resume_started,
                }
            report["cases"].append(case)
            report["total_wall_seconds_so_far"] = time.perf_counter() - started
            atomic_json(report_path, report)
            del model, optimizer, first, loaders
            torch.cuda.empty_cache()
    report["total_wall_seconds"] = time.perf_counter() - started
    report["finished_at_utc"] = datetime.now(UTC).isoformat()
    atomic_json(report_path, report)
    return report
