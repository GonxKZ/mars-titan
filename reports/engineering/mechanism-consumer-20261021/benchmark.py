"""Coste del consumidor provisional con fixtures y CUDA, sin ajuste ni corpus real."""

import argparse
import hashlib
import io
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/models/titans"))

from test_context_selection import batch_on  # noqa: E402
from test_episodic_readout import CODEC, CONTEXT, source  # noqa: E402
from test_market_memory import KINDS  # noqa: E402
from test_mechanism_consumer import calibration, consumer  # noqa: E402

from mars_titan.calibration.cohort_feedback import CohortOutcome  # noqa: E402
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot  # noqa: E402
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer  # noqa: E402
from mars_titan.models.titans.mechanism_catalog import MechanismAction  # noqa: E402

CASES = (
    "historical_consumer",
    "disabled",
    "reader",
    "context_static",
    "context_online",
    "byte_return",
)


def fixture(kind, rows):
    context = kind.startswith("context") or kind == "byte_return"
    engine = consumer(
        "cuda:0",
        kind=KINDS[0] if context else None,
        reader=kind in ("reader", "context_online", "byte_return"),
        populated=kind in ("reader", "context_online", "byte_return"),
        calibrator=calibration(max_bytes=4096 if kind == "byte_return" else 8 * 1024**2)
        if context
        else None,
        action=MechanismAction(
            "parent" if context else "off",
            "k4cr" if kind in ("reader", "context_online", "byte_return") else "k1",
            "static" if kind == "context_static" else "online" if context else "none",
        ),
    )
    flows = tuple(f"{market}/A{i:03d}" for market in ("CN", "US") for i in range(rows // 2))
    episodes = source(64) if engine.readout is not None else None
    historical = (
        FrozenFinancialConsumer(engine.predictor) if kind == "historical_consumer" else None
    )
    return engine, flows, episodes, historical


def trajectory(kind, fixture_):
    engine, flows, episodes, historical = fixture_
    predictor = engine.predictor
    auxiliary = engine.initial_state() if historical is None else None
    fast = predictor.initial_state(flows)
    emissions, counters, state_sizes = [], [], []
    for event in range(3):
        batch = batch_on(predictor, flows, event)
        at = batch.prediction_at[0]
        snapshot = (
            EpisodeSnapshot.create(
                **episodes,
                cutoff=at,
                codec_id=CODEC,
                context_id=CONTEXT,
                dtype=torch.float64,
                device="cuda:0",
            )
            if episodes is not None and event
            else None
        )
        if historical is not None:
            result = historical.prepare(batch, fast, context_id=CONTEXT, warmup=event == 0)
            fast = result.next_state
            if event:
                emissions.append(result.quantiles.detach().cpu().numpy())
        else:
            proposal = engine.begin(
                auxiliary,
                flow_ids=flows,
                cutoff=at,
                context_id=CONTEXT,
                matures_at=(at + 10,) * len(flows) if event else None,
                snapshot=snapshot if event else None,
                warmup=event == 0,
            )
            block = engine.prepare(batch, fast, proposal)
            result = engine.finish(block.event)
            auxiliary, fast = result.state, block.next_state
            counters.append(dict(result.work))
            if event:
                emissions.append(result.quantiles)
            if event == 1:
                previous = result
            if event == 2 and auxiliary.calibration is not None:
                tracked = {
                    item["prediction_at"] for item in auxiliary.calibration.export()["cohorts"]
                }
                outcomes = tuple(
                    CohortOutcome(
                        row["flow_id"].split("/")[0],
                        previous.state.cutoff,
                        row["sample_id"],
                        at,
                        float(i) / len(flows),
                    )
                    for i, row in enumerate(previous.rows)
                    if previous.state.cutoff in tracked
                )
                auxiliary, _ = engine.feedback(auxiliary, outcomes, at=at)
        payload = dict(fast=predictor.export_state_cpu(fast))
        if historical is None:
            payload["auxiliary"] = engine.export_state(auxiliary)
            auxiliary = engine.restore_state(payload["auxiliary"])
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        state_sizes.append(buffer.tell())
    return emissions, counters, state_sizes


def percentiles(values):
    values = np.asarray(values)
    return dict(
        mean=float(values.mean()),
        std=float(values.std(ddof=1)),
        **{f"p{p}": float(np.percentile(values, p)) for p in (50, 95, 99)},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.repetitions < 2 or args.warmup < 1:
        raise ValueError("La salida debe ser nueva y las repeticiones deben ser válidas")
    if not torch.cuda.is_available():
        raise RuntimeError("Esta medida necesita CUDA y no admite sustitución por CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        device="cuda:0",
        gpu=torch.cuda.get_device_name(0),
        torch=torch.__version__,
        python=platform.python_version(),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        precision="fp64_strict",
        hidden_size=32,
        price_context=64,
        episodes=64,
        cpu_threads=torch.get_num_threads(),
        warmup=args.warmup,
        repetitions=args.repetitions,
        scientific_study=False,
        optimizer_steps=0,
        real_data_used=False,
        durable_publication=False,
        energy_joules=None,
        cases=[],
    )
    baselines = {}
    for rows in (64, 256):
        for kind in CASES:
            init = time.perf_counter()
            current = fixture(kind, rows)
            report.setdefault("implementation", current[0].identity()["implementation"])
            torch.cuda.synchronize()
            initialized_ms = (time.perf_counter() - init) * 1000
            samples = []
            torch.cuda.reset_peak_memory_stats()
            for repetition in range(args.warmup + args.repetitions):
                torch.cuda.synchronize()
                before = time.perf_counter_ns()
                predictions, work, sizes = trajectory(kind, current)
                torch.cuda.synchronize()
                elapsed = (time.perf_counter_ns() - before) / 1e6
                if kind == "historical_consumer":
                    baselines[rows] = predictions
                elif kind == "disabled":
                    for actual, expected in zip(predictions, baselines[rows], strict=True):
                        np.testing.assert_array_equal(actual, expected)
                if repetition >= args.warmup:
                    samples.append(elapsed)
            report["cases"].append(
                dict(
                    rows=rows,
                    kind=kind,
                    initialization_ms=initialized_ms,
                    trajectory_ms=percentiles(samples),
                    raw_samples_ms=samples,
                    session_count=3,
                    predicted_sessions=2,
                    observations=rows * 3,
                    observations_per_second=rows * 3 * 1000 / float(np.mean(samples)),
                    allocated_vram_peak_bytes=torch.cuda.max_memory_allocated(),
                    reserved_vram_peak_bytes=torch.cuda.max_memory_reserved(),
                    serialized_proposal_bytes=sizes,
                    work=work,
                )
            )
            del current
    report["internal_seconds"] = time.perf_counter() - started
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["scope"] = (
        "Tres sesiones, una de calentamiento y dos de predicción, con fixtures de ambos mercados. "
        "Incluye preparación de entradas CPU, transferencia a CUDA, estados iniciales, auditorías, "
        "MAC, contexto, lector, calibración, feedback manual, exportación, restauración auxiliar "
        "y serialización en RAM. No incluye escritura atómica, métricas financieras ni disco. "
        "Las comprobaciones de paridad quedan fuera de cada tiempo y dentro del proceso. "
        "RSS corresponde al proceso completo. La VRAM se mide por caso e incluye el calentamiento "
        "y la memoria reservada por PyTorch. Los percentiles son empíricos y tienen pocas muestras."
    )
    sources = [Path(__file__)]
    sources += [
        ROOT / "src/mars_titan/models/titans" / name
        for name in (
            "mechanism_catalog.py",
            "mechanism_consumer.py",
            "mechanism_event.py",
            "mechanism_calibration.py",
        )
    ]
    sources += [
        ROOT / "tests/models/titans" / name
        for name in (
            "test_mechanism_consumer.py",
            "test_context_selection.py",
            "test_market_memory.py",
            "test_memory_stability.py",
            "test_financial_adapter.py",
            "test_episodic_readout.py",
        )
    ]
    report["sources"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sources
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(report["cases"]), seconds=report["internal_seconds"])))


if __name__ == "__main__":
    main()
