"""Coste de fases congeladas sintéticas, sin aprendizaje ni datos financieros reales."""

import hashlib
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import mars_titan
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.financial_observations import ObservationEvent
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.memory.spaced_retention import SpacedRetentionConfig
from mars_titan.training.mars_titan_run import MarsTitanInference, ReadoutRecipe
from mars_titan.training.spaced_inference import SpacedInference

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]
from test_financial_session import moment  # noqa: E402
from test_financial_session_controls import (  # noqa: E402
    flow_source,
    no_target_estimation,
    paired_consumers,
)
from test_mars_titan_session_parity import PHASE, raw  # noqa: E402


def sequence(source, flows):
    previous, result = [], []
    for at in (125, 126, 127, 128, 129, 130):
        inputs = () if at == 129 else (raw(source["batches"][at]),)
        labels = tuple((flow, decided, 0.1) for flow, decided in previous)
        result.append(ObservationEvent(moment(at), inputs, labels, False))
        previous = [(flow, moment(at)) for flow in flows] if inputs else []
    return [*result, ObservationEvent(moment(201), (), (), True)]


def trial(native, source, consumer, events, active, capacity):
    recipe = ReadoutRecipe(
        loss="mae",
        neighbors=4,
        block_rows=256,
        max_working_bytes=consumer.readout.config.max_working_bytes,
    )
    common = dict(native=native, codec=source["codec"], world="spaced_cost", fold="0", audit=True)
    if active:
        engine = SpacedInference(
            consumer,
            recipe,
            prefixes=source["prefixes"],
            retention=SpacedRetentionConfig(policy="reservoir", capacity=capacity, frontier=1),
            **common,
        )
    else:
        engine = MarsTitanInference(
            consumer.predictor,
            consumer.readout,
            recipe,
            admission="m1",
            retention=RetentionConfig(capacity=capacity, frontier=1),
            **common,
        )
    started = time.perf_counter()
    metrics = engine._pass(PHASE, events)
    elapsed = time.perf_counter() - started
    points = [event[2:5] for event in engine.audit if event[0] == "prediction"]
    common_metrics = {key: value for key, value in metrics.items() if key != "spaced"}
    signature = hashlib.sha256(
        json.dumps([points, common_metrics], sort_keys=True, allow_nan=False).encode()
    ).hexdigest()
    return dict(seconds=elapsed, common_sha256=signature, predictions=len(points))


def main():
    assert Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    assert not torch.cuda.is_initialized()
    native = load_native()
    guard = no_target_estimation.__wrapped__()
    next(guard)

    def reject_optimizer(*_):
        raise RuntimeError("La medida técnica no admite optimizadores")

    handle = register_optimizer_step_pre_hook(reject_optimizer)
    started, rows = time.perf_counter(), []
    report = dict(
        kind="technical_frozen_spaced_window_cost",
        status="running",
        hardware=dict(cpu=cpu_name(), platform=platform.platform()),
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            device="cpu",
            cpu_threads=2,
            native_sha256=native.binary_sha256,
        ),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation={
            name: hashlib.sha256(Path(sys.modules[name].__file__).read_bytes()).hexdigest()
            for name in (
                "mars_titan.training.spaced_inference",
                "mars_titan.training.mars_titan_run",
                "mars_titan.memory.spaced_session",
                "mars_titan.memory.spaced_utility",
                "mars_titan.memory.spaced_retention",
            )
        },
        shapes=dict(hidden=32, neighbors=4, dtype="float64", dimensions=[5, 2, 1, 3, 3]),
        measured_scope="frozen_phase_including_MAC_readout_ledger_admission_and_closure",
        excluded=[
            "imports",
            "fixture_preparation",
            "construction",
            "durable_storage",
            "dataset_reads",
            "GPU",
            "energy",
        ],
        includes_integrity_checks=True,
        warmup=1,
        measured_repetitions=5,
        samples=rows,
    )
    try:
        for assets in (64, 256):
            flows = tuple(f"US/F{i:04d}" for i in range(assets))
            source = flow_source(out / f"fixture-{assets}", flows=flows)
            consumer = paired_consumers(source)["disabled"]
            events = sequence(source, flows)
            capacity = min(1024, 4 * assets)
            for repetition in range(6):
                pair = []
                for active in (False, True) if repetition % 2 == 0 else (True, False):
                    value = trial(native, source, consumer, events, active, capacity)
                    value.update(
                        assets=assets,
                        capacity=capacity,
                        active=active,
                        repetition=repetition,
                        warmup=repetition == 0,
                    )
                    rows.append(value)
                    pair.append(value)
                assert pair[0]["common_sha256"] == pair[1]["common_sha256"]
                atomic_json(out / "cost.json", report)
        report["summaries"] = [
            dict(
                assets=assets,
                active=active,
                median_seconds=statistics.median(values),
                minimum_seconds=min(values),
                maximum_seconds=max(values),
                samples=len(values),
            )
            for assets in (64, 256)
            for active in (False, True)
            for values in [
                [
                    row["seconds"]
                    for row in rows
                    if row["assets"] == assets and row["active"] == active and not row["warmup"]
                ]
            ]
        ]
        report.update(
            status="completed",
            process_seconds_without_imports=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_initialized=torch.cuda.is_initialized(),
        )
        atomic_json(out / "cost.json", report)
        print(
            json.dumps(
                dict(status=report["status"], samples=len(rows), summaries=report["summaries"])
            )
        )
    finally:
        handle.remove()
        guard.close()


if __name__ == "__main__":
    main()
