"""Medir lectura histórica sobre sesiones de prueba, sin ajustar parámetros."""

import argparse
import hashlib
import json
import platform
import resource
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]

from test_financial_mechanisms import engines  # noqa: E402
from test_financial_session import moment  # noqa: E402
from test_financial_session_controls import flow_source  # noqa: E402

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession  # noqa: E402
from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig  # noqa: E402
from mars_titan.memory.retention_bank import RetentionConfig  # noqa: E402

MODES = ("receipt_latest", "asof_complete", "asof_partial")


def measure(run, mode):
    if mode == MODES[0]:
        return run.mechanism_collection(1)
    return run.mechanism_collection_at(
        1,
        through_generation=3 if mode == MODES[1] else 2,
        before=moment(132) if mode == MODES[1] else moment(128),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("La salida necesita una ruta nueva dentro de un directorio existente")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0, sin alternativa CPU implícita")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = load_native()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        device="cuda:0",
        dtype="torch.float64",
        cpu_threads=torch.get_num_threads(),
        real_corpus=False,
        optimizer_steps=0,
        warmup=3,
        repetitions=20,
        cases=[],
        native_binary_sha256=hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mars-asof-cost-") as folder:
        root = Path(folder)
        for count in (4, 64, 256):
            source = flow_source(
                root / f"source-{count}", flows=tuple(f"US/T{i:04d}" for i in range(count))
            )
            engine = engines(source, "cuda:0", online=True)[3]
            setup_start = time.perf_counter()
            with FinancialSession(
                root / f"session-{count}",
                native=native,
                consumer=engine,
                codec=source["codec"],
                prefixes=source["prefixes"],
                retention=RetentionConfig(capacity=256, frontier=8, new_candidates=8),
                phase=FinancialPhase(
                    "validation", moment(125), moment(126), moment(130), moment(131)
                ),
                admission="m1",
                block_rows=64,
                world="asof_cost_fixture",
                fold="0",
                mechanism_collection=MechanismCollectionConfig(),
            ) as run:
                issued = run.step(source["batches"][126], [])
                labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
                run.step([], labels[: count // 2], kind="settlement", cutoff=moment(127))
                prefix = measure(run, "asof_partial")
                run.step(
                    [],
                    labels[count // 2 :],
                    kind="settlement",
                    cutoff=moment(131),
                    close_phase=True,
                )
                setup_seconds = time.perf_counter() - setup_start
                reference = measure(run, "receipt_latest")
                complete = measure(run, "asof_complete")
                before = run.snapshot()
                rng, gpu_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state(0)
                samples = {mode: [] for mode in MODES}
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                resident = torch.cuda.memory_allocated(0)
                source_bytes = {}
                for repetition in range(23):
                    for mode in MODES[repetition % 3 :] + MODES[: repetition % 3]:
                        torch.cuda.synchronize(0)
                        tick = time.perf_counter_ns()
                        result = measure(run, mode)
                        torch.cuda.synchronize(0)
                        elapsed = (time.perf_counter_ns() - tick) / 1e6
                        if mode == "asof_partial":
                            if result != prefix:
                                raise RuntimeError(
                                    "Las recepciones posteriores cambiaron el prefijo"
                                )
                        else:
                            if {k: result[k] for k in reference} != reference:
                                raise RuntimeError(
                                    "La reconstrucción completa difiere del recibo confirmado"
                                )
                        if mode != "receipt_latest":
                            source_bytes[mode] = result["as_of"]["source_bytes"]
                        if repetition >= 3:
                            samples[mode].append(elapsed)
                if (
                    run.snapshot() != before
                    or not torch.equal(torch.random.get_rng_state(), rng)
                    or not torch.equal(torch.cuda.get_rng_state(0), gpu_rng)
                ):
                    raise RuntimeError("La lectura cambió la sesión o sus generadores aleatorios")
                for mode, values in samples.items():
                    report["cases"].append(
                        dict(
                            rows=count,
                            mode=mode,
                            samples_ms=values,
                            p50_ms=float(np.percentile(values, 50)),
                            p95_ms=float(np.percentile(values, 95)),
                            p99_ms=float(np.percentile(values, 99)),
                            calls_per_second=1000 / float(np.mean(values)),
                            packet_and_native_bytes=source_bytes.get(mode),
                            setup_seconds=setup_seconds,
                            allocated_resident_bytes=resident,
                            allocated_peak_bytes=torch.cuda.max_memory_allocated(0),
                            reserved_peak_bytes=torch.cuda.max_memory_reserved(0),
                            provenance=dict(
                                contract_id=run.contract_id,
                                model_id=run.model_id,
                                consumer_id=engine.model_id,
                                parameter_id=engine.predictor._parameter_id,
                                input_spec_id=source["spec"].fingerprint(),
                                collection_sha256=prefix["collection"]["sha256"],
                                native_records=complete["as_of"]["native_records"],
                                phase=asdict(run.phase),
                                supervision_origin=source["label_origin"],
                            ),
                        )
                    )
                print(json.dumps(dict(rows=count, cases=report["cases"][-3:])), flush=True)
    report.update(
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "src/mars_titan/memory/mechanism_collection_asof.py",
                ROOT / "src/mars_titan/memory/financial_mechanisms.py",
                ROOT / "src/mars_titan/memory/financial_session.py",
            )
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
