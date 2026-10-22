"""Comparar la caché de control con fases nativas ficticias y el mismo modelo CUDA."""

import argparse
import hashlib
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import torch

from mars_titan.memory.financial_session import FinancialSession
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.write_policy import CompositeScoreConfig, MatureErrorConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

ROOT = Path.cwd()


def fixture_components():
    for folder in ("tests/models/titans", "tests/memory", "tests/calibration"):
        sys.path.insert(0, str(ROOT / folder))
    from test_financial_session_controls import flow_source
    from test_mars_titan_session_parity import SCALERS, varied
    from test_mechanism_admission import PHASE, configured, feedback

    return flow_source, SCALERS, varied, PHASE, configured, feedback


flow_source, SCALERS, varied, PHASE, configured, feedback = fixture_components()

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
OUT = args.output
OUT.mkdir(parents=True, exist_ok=False)
if not torch.cuda.is_available():
    raise RuntimeError("Estas medidas requieren CUDA y no admiten pasar a CPU")
torch.cuda.set_device("cuda:0")
torch.backends.mha.set_fastpath_enabled(False)
native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
started = time.perf_counter()


def gpu():
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.used,utilization.gpu",
            "--format=csv",
        ],
        text=True,
    )


def digest(value):
    content = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(content).hexdigest()


report = dict(
    recorded_at_utc=datetime.now(UTC).isoformat(),
    device="cuda:0",
    dtype="torch.float64",
    gpu=torch.cuda.get_device_name(0),
    gpu_before=gpu(),
    torch=str(torch.__version__),
    cuda=torch.version.cuda,
    python=sys.version,
    capacity=16,
    hidden_size=32,
    physical_block_rows=64,
    reader_budget_bytes=128 * 1024**2,
    cache_budget_bytes=8 * 1024**2,
    warmup=2,
    repetitions=3,
    maximum_seconds=1200,
    cases=[],
    incomplete=[],
    processes_before=subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
        text=True,
    ),
)


def run(path, source, consumer, admission, enabled):
    retention = (
        MatureErrorConfig(capacity=16)
        if admission == "m2"
        else CompositeScoreConfig(SCALERS, capacity=16)
    )
    prior, points, peak_charge = [], {}, 0
    with FinancialSession(
        path,
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=retention,
        phase=PHASE,
        world="control_cache_benchmark_fixture",
        fold="0",
        admission=admission,
        block_rows=64,
    ) as session:
        session.artifacts.max_cached_bytes = report["cache_budget_bytes"] if enabled else 0
        session.step(source["batches"][125], [], kind="warmup")
        for index in (126, 127, 128, 130):
            prior = session.step(
                source["batches"][index], feedback(native, prior, index)
            ).predictions
            points.update({(p.asset, p.decision_at): p.value for p in prior})
            cached = session.artifacts._control_cache
            peak_charge = max(peak_charge, cached[1] if cached is not None else 0)
        session.step([], [], kind="settlement", cutoff=PHASE.close_at, close_phase=True)
        snapshot = session.snapshot()
        n = len(source["batches"][125])
        assert snapshot["issued"] == 4 * n and snapshot["applied"] == 3 * n
    return points, digest(snapshot), peak_charge


def measure(path, source, consumer, admission, enabled):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    initial = torch.cuda.memory_allocated()
    before = time.perf_counter()
    points, state, charge = run(path, source, consumer, admission, enabled)
    torch.cuda.synchronize()
    sample = dict(
        milliseconds=1000 * (time.perf_counter() - before),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        incremental_peak_bytes=torch.cuda.max_memory_allocated() - initial,
        peak_control_cache_charge_bytes=charge,
        final_snapshot_sha256=state,
    )
    return points, sample


sources = {}
with torch.no_grad():
    for assets, admission, automatic in (
        (4, "m3", False),
        (64, "m2", False),
        (64, "m3", False),
        (64, "m3", True),
        (256, "m3", False),
    ):
        name = f"{admission}-{assets}-" + ("manual-k2c" if automatic else "fixed-k4cc")
        if time.perf_counter() - started > report["maximum_seconds"]:
            report["incomplete"].append(name)
            continue
        if assets not in sources:
            sources[assets] = varied(
                flow_source(
                    OUT / f"fixture-{assets}", flows=tuple(f"US/T{i:04d}" for i in range(assets))
                )
            )
        source = sources[assets]
        assert source["label_origin"] == "manual_fixture_no_estimation"
        parent = configured(source, "cuda:0", automatic=automatic)[3]
        config = replace(
            parent.readout.config,
            max_working_bytes=report["reader_budget_bytes"],
            evidence_bridge=replace(
                parent.readout.config.evidence_bridge,
                max_working_bytes=report["reader_budget_bytes"],
            ),
        )
        consumer = FrozenMechanismConsumer(
            parent.predictor,
            readout=EpisodicReadout(config, device="cuda:0", dtype=torch.float64)
            .eval()
            .requires_grad_(False),
            catalog=parent.catalog,
            **(dict(policy=parent.policy) if automatic else dict(action=parent.action)),
        )
        for warmup in range(2):
            for enabled in (False, True):
                measure(
                    OUT / "runs" / name / f"warm{warmup}-{int(enabled)}",
                    source,
                    consumer,
                    admission,
                    enabled,
                )
        samples, max_difference = {"uncached": [], "cached": []}, 0.0
        for repetition in range(3):
            outputs = {}
            for enabled in (False, True) if repetition % 2 == 0 else (True, False):
                method = "cached" if enabled else "uncached"
                outputs[method], sample = measure(
                    OUT / "runs" / name / f"rep{repetition}-{int(enabled)}",
                    source,
                    consumer,
                    admission,
                    enabled,
                )
                samples[method].append(sample)
            assert outputs["cached"].keys() == outputs["uncached"].keys()
            for key, value in outputs["cached"].items():
                reference = outputs["uncached"][key]
                assert math.isfinite(value) and value == reference
                max_difference = max(max_difference, abs(value - reference))
            assert (
                samples["cached"][-1]["final_snapshot_sha256"]
                == samples["uncached"][-1]["final_snapshot_sha256"]
            )
        report["cases"].append(
            dict(
                name=name,
                assets=assets,
                admission=admission,
                manual_policy=automatic,
                max_prediction_abs_difference=max_difference,
                exact_snapshot_parity=True,
                samples=samples,
                median_ms={
                    key: statistics.median(s["milliseconds"] for s in rows)
                    for key, rows in samples.items()
                },
            )
        )
        (OUT / "partial.json").write_text(json.dumps(report, indent=2) + "\n")
        print(name, report["cases"][-1]["median_ms"], flush=True)

report.update(
    total_process_seconds=time.perf_counter() - started,
    process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    native_binary_sha256=native.binary_sha256,
    gpu_after=gpu(),
    source_sha256={
        path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in (
            "src/mars_titan/memory/session_artifacts.py",
            "src/mars_titan/memory/financial_session.py",
            "src/mars_titan/memory/financial_mechanisms.py",
            "tests/memory/test_mechanism_admission.py",
            "tests/memory/test_financial_session_controls.py",
            "tests/memory/test_mars_titan_session_parity.py",
        )
    },
    benchmark_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    scope=(
        "Fases técnicas con cinco observaciones, cuatro decisiones y cierre. "
        "Dos calentamientos y tres pares por caso, orden alternado. "
        "Sin corpus real, optimizador, ajuste ni acceso a 2024."
    ),
    memory_scope=(
        "VRAM son asignaciones de PyTorch. RAM es pico global del proceso. "
        "El cargo de caché usa la convención del validador más archivo y margen, "
        "no el heap exacto de Python."
    ),
    concurrency_scope=(
        "Otro modelo local seguía en la GPU. No había otras pruebas, mutaciones "
        "o consultas propias en paralelo."
    ),
    optimizer_steps=0,
    real_corpus=False,
    final_test_opened=False,
)
(OUT / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")
assert not report["incomplete"], report["incomplete"]
