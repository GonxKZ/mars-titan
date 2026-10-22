"""Coste CUDA de sesiones financieras selladas con fixtures, sin ajustar parámetros."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]

from test_financial_mechanisms import engines  # noqa: E402
from test_financial_session import moment  # noqa: E402
from test_financial_session_controls import flow_source  # noqa: E402
from test_mechanism_consumer import calibration  # noqa: E402

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession  # noqa: E402
from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.memory.retention_bank import RetentionConfig  # noqa: E402
from mars_titan.models.titans.episodic_readout import (  # noqa: E402
    EpisodicReadout,
    EpisodicReadoutConfig,
)
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer  # noqa: E402
from mars_titan.models.titans.mechanism_catalog import MechanismAction  # noqa: E402
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer  # noqa: E402

CASES = ("previous", "disabled", "context_online", "context_reader_online")


def consumer(source, kind):
    context = kind.startswith("context")
    data = engines(source, "cuda:0", online=context)
    engine = data[3]
    if kind == "previous":
        return FrozenFinancialConsumer(engine.predictor)
    if kind == "context_reader_online":
        reader = (
            EpisodicReadout(
                EpisodicReadoutConfig(source["codec"].fingerprint(), hidden_size=32, refinements=1),
                device="cuda:0",
                dtype=torch.float64,
            )
            .eval()
            .requires_grad_(False)
        )
        return FrozenMechanismConsumer(
            engine.predictor,
            readout=reader,
            calibration=calibration(),
            action=MechanismAction("parent", "k4cr", "online"),
        )
    return engine


def trajectory(native, source, engine, output):
    cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(0)
    times, predictions = {}, []
    options = dict(
        native=native,
        consumer=engine,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=RetentionConfig(capacity=64, frontier=2, new_candidates=4),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
        admission="m1",
        block_rows=64,
        world="fixture_mechanism_cost",
        fold="0",
    )
    start = time.perf_counter_ns()
    run = FinancialSession(output, **options)
    torch.cuda.synchronize(0)
    times["open"] = (time.perf_counter_ns() - start) / 1e6
    prior = []
    with run:
        for index, kind in (
            (125, "warmup"),
            (126, "decision"),
            (127, "decision"),
            (128, "decision"),
        ):
            available_at = source["batches"][index][0].prediction_at[0]
            labels = [native.Feedback(value.id, 0, available_at, 0.125) for value in prior]
            before = time.perf_counter_ns()
            result = run.step(source["batches"][index], labels, kind=kind)
            torch.cuda.synchronize(0)
            times[f"step_{index}"] = (time.perf_counter_ns() - before) / 1e6
            prior = result.predictions
            predictions.extend((value.asset, value.decision_at, value.value) for value in prior)
        before = time.perf_counter_ns()
        run.step([], [], kind="settlement", cutoff=options["phase"].close_at, close_phase=True)
        torch.cuda.synchronize(0)
        times["close_phase"] = (time.perf_counter_ns() - before) / 1e6
        expected = run.snapshot()
        if type(engine) is FrozenMechanismConsumer:
            event = run.mechanism_event(2)
            assert len(event["payload"]["rows"]) == len(source["batches"][126])
        bytes_ = sum(path.stat().st_size for path in Path(output).rglob("*") if path.is_file())
    before = time.perf_counter_ns()
    with FinancialSession(output, **options, resume=True) as restored:
        assert restored.snapshot() == expected
        if type(engine) is FrozenMechanismConsumer:
            assert restored.mechanism_event(2) == event
    torch.cuda.synchronize(0)
    times["recover"] = (time.perf_counter_ns() - before) / 1e6
    times["total"] = (time.perf_counter_ns() - start) / 1e6
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
    return times, bytes_, hashlib.sha256(json.dumps(predictions).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.storage_root.is_dir()
        or args.repetitions < 2
        or args.warmup < 1
        or any(rows not in (4, 64, 256) for rows in args.rows)
    ):
        raise ValueError("El coste necesita salida nueva, almacenamiento y tamaños acotados")
    if not torch.cuda.is_available():
        raise RuntimeError("Esta medida requiere CUDA, sin pasar a CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    started = time.perf_counter()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        device="cuda:0",
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        numpy=np.__version__,
        cpu_threads=torch.get_num_threads(),
        dtype="float64",
        block_rows=64,
        bank_capacity=64,
        modalities=dict(prices=[64, 5], news=2, charts=1, fundamentals=3, macro=3),
        real_data_used=False,
        optimizer_steps=0,
        scientific_study=False,
        warmup=args.warmup,
        repetitions=args.repetitions,
        native_sha256=native.binary_sha256,
        cases=[],
    )
    with tempfile.TemporaryDirectory(
        prefix="mars-session-cost-", dir=args.storage_root
    ) as directory:
        folder = Path(directory)
        for rows in args.rows:
            source = flow_source(
                folder / f"source-{rows}", flows=tuple(f"US/T{i:04d}" for i in range(rows))
            )
            reference_digest = None
            for kind in CASES:
                engine = consumer(source, kind)
                samples, sizes, digests = [], [], []
                torch.cuda.reset_peak_memory_stats(0)
                for repetition in range(args.warmup + args.repetitions):
                    values, size, digest = trajectory(
                        native, source, engine, folder / f"run-{rows}-{kind}-{repetition}"
                    )
                    if repetition >= args.warmup:
                        samples.append(values)
                        sizes.append(size)
                        digests.append(digest)
                assert len(set(digests)) == 1
                if kind == "previous":
                    reference_digest = digests[0]
                if kind == "disabled":
                    assert digests[0] == reference_digest
                totals = [sample["total"] for sample in samples]
                report["cases"].append(
                    dict(
                        rows=rows,
                        kind=kind,
                        model_id=engine.model_id,
                        consumer_identity=engine.identity(),
                        prefix_policy_id=source["prefixes"].policy_id,
                        samples_ms=samples,
                        **{f"p{p}_ms": float(np.percentile(totals, p)) for p in (50, 95, 99)},
                        mean_ms=float(np.mean(totals)),
                        std_ms=float(np.std(totals, ddof=1)),
                        prediction_rows_per_second=3 * rows * 1000 / float(np.mean(totals)),
                        retained_bytes=sizes,
                        prediction_sha256=digests[0],
                        allocated_vram_peak_bytes=torch.cuda.max_memory_allocated(0),
                        reserved_vram_peak_bytes=torch.cuda.max_memory_reserved(0),
                    )
                )
                print(json.dumps(dict(rows=rows, kind=kind, completed=True)), flush=True)
    report["process_seconds"] = time.perf_counter() - started
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["scope"] = (
        "Fixtures US con precios replicados, máscaras y etiquetas manuales. Incluye apertura, "
        "un calentamiento, tres emisiones, dos recepciones de resultados, cierre censurado, "
        "escritura sellada, verificación, poda y recuperación. Total incluye consultas de "
        "trazas y contabilidad de archivos. La construcción de datos y modelos queda fuera "
        "de las latencias y dentro del proceso. Dimensiones pequeñas de modalidades, sin "
        "codificadores externos ni corpus real. Sin energía medida. Percentiles empíricos "
        "y orden fijo de versiones."
    )
    report["sources"] = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            ROOT / "src/mars_titan/memory/financial_mechanisms.py",
            ROOT / "src/mars_titan/memory/financial_session.py",
            ROOT / "src/mars_titan/memory/financial_consumers.py",
            ROOT / "src/mars_titan/models/titans/frozen_financial.py",
            ROOT / "src/mars_titan/models/titans/mechanism_consumer.py",
            ROOT / "tests/memory/test_financial_mechanisms.py",
            ROOT / "tests/memory/test_financial_session_controls.py",
        )
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(report["cases"]), seconds=report["process_seconds"])))


if __name__ == "__main__":
    main()
