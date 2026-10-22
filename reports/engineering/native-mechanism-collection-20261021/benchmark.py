"""Coste técnico de una sesión nativa con y sin recogida, sobre fixtures manuales."""

import argparse
import hashlib
import json
import platform
import resource
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

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession  # noqa: E402
from mars_titan.memory.mechanism_collection import state_digest  # noqa: E402
from mars_titan.memory.mechanism_collection_record import CollectedMechanisms  # noqa: E402
from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig  # noqa: E402
from mars_titan.memory.retention_bank import RetentionConfig  # noqa: E402
from mars_titan.models.titans.episodic_readout import (  # noqa: E402
    EpisodicReadout,
    EpisodicReadoutConfig,
)
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer  # noqa: E402

MODES = ("native_fixed", "native_collect")


def run_case(native, source, engine, output, mode):
    options = dict(mechanism_collection=MechanismCollectionConfig()) if mode == MODES[1] else {}
    with FinancialSession(
        output,
        native=native,
        consumer=engine,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=RetentionConfig(capacity=256, frontier=8, new_candidates=8),
        phase=FinancialPhase("validation", moment(125), moment(126), moment(130), moment(131)),
        admission="m1",
        block_rows=64,
        world="native_collection_cost_fixture",
        fold="0",
        **options,
    ) as run:
        torch.cuda.synchronize(0)
        torch.cuda.reset_peak_memory_stats(0)
        started = time.perf_counter_ns()
        run.step(source["batches"][125], [], kind="warmup")
        issued = run.step(source["batches"][126], [])
        labels = [native.Feedback(p.id, 0, moment(127), 0.125) for p in issued.predictions]
        run.step(source["batches"][127], labels)
        following = run.step(source["batches"][128], [])
        run.step([], [], kind="settlement", cutoff=moment(131), close_phase=True)
        torch.cuda.synchronize(0)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        allocated, reserved = torch.cuda.max_memory_allocated(0), torch.cuda.max_memory_reserved(0)
        points = np.array([p.value for p in (*issued.predictions, *following.predictions)])
        fast = run._fast_store.gather(
            run.fast_state(), tuple(sorted(run.fast_state()["references"]))
        )
        fast_sha = state_digest(engine.predictor.export_state(fast))
        packet_bytes = 0
        observed = 0
        if mode == MODES[1]:
            packet = CollectedMechanisms.restore(run.mechanism_collection(4)["collection"])
            packet_bytes = len(packet.record()["payload_utf8"])
            observed = sum(
                c["status"] == "observed" for c in packet.payload()["groups"][0]["candidates"]
            )
            if observed != 36:
                raise RuntimeError("La medida necesita las36 alternativas tras el prefijo poblado")
        diagnostics = run.diagnostics()
        disk = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
        disk += sum(p.stat().st_size for p in (output / "artifacts").iterdir() if p.is_file())
        return dict(
            elapsed_ms=elapsed,
            allocated_peak_bytes=allocated,
            reserved_peak_bytes=reserved,
            artifact_and_native_bytes=disk,
            last_packet_bytes=packet_bytes,
            observed_candidates=observed,
            diagnostics=diagnostics,
            fast_sha256=fast_sha,
        ), points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("La salida necesita una ruta nueva en un directorio existente")
    if args.rows not in ([64, 256], [4]) or not 2 <= args.repetitions <= 4:
        raise ValueError("Tamaños o repeticiones fuera del protocolo acotado")
    if not torch.cuda.is_available():
        raise RuntimeError("Se requiere cuda:0 y no se pasa a CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = load_native()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        python=platform.python_version(),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        device="cuda:0",
        dtype="torch.float64",
        cpu_threads=torch.get_num_threads(),
        real_corpus=False,
        optimizer_steps=0,
        warmup=1,
        repetitions=args.repetitions,
        cases=[],
    )
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mars-native-collection-cost-") as folder:
        root = Path(folder)
        for size in args.rows:
            source = flow_source(
                root / f"source-{size}", flows=tuple(f"US/T{i:04d}" for i in range(size))
            )
            parent = engines(source, "cuda:0", online=True)[3]
            reader = (
                EpisodicReadout(
                    EpisodicReadoutConfig(source["codec"].fingerprint(), hidden_size=32),
                    device="cuda:0",
                    dtype=torch.float64,
                )
                .eval()
                .requires_grad_(False)
            )
            engine = FrozenMechanismConsumer(
                parent.predictor,
                readout=reader,
                calibration=parent.initial_state().calibration,
                action=parent.action,
            )
            reference = None
            reference_fast = None
            reference_counts = None
            cases = {mode: [] for mode in MODES}
            differences = dict.fromkeys(MODES, 0.0)
            for repetition in range(args.repetitions + 1):
                for mode in MODES[repetition % 2 :] + MODES[: repetition % 2]:
                    case, points = run_case(
                        native, source, engine, root / f"{size}-{repetition}-{mode}", mode
                    )
                    counts = case["diagnostics"]
                    if reference is None:
                        reference, reference_fast, reference_counts = (
                            points.copy(),
                            case["fast_sha256"],
                            counts,
                        )
                    np.testing.assert_allclose(points, reference, rtol=1e-12, atol=1e-12)
                    if case["fast_sha256"] != reference_fast or counts != reference_counts:
                        raise RuntimeError(
                            "La recogida cambió el estado rápido o los recuentos financieros"
                        )
                    differences[mode] = max(
                        differences[mode], float(np.max(np.abs(points - reference)))
                    )
                    print(
                        json.dumps(
                            dict(
                                rows=size,
                                repetition=repetition,
                                mode=mode,
                                elapsed_ms=case["elapsed_ms"],
                            )
                        ),
                        flush=True,
                    )
                    if repetition:
                        cases[mode].append(case)
            for mode in MODES:
                samples = np.array([c["elapsed_ms"] for c in cases[mode]])
                report["cases"].append(
                    dict(
                        rows=size,
                        mode=mode,
                        samples=cases[mode],
                        p50_ms=float(np.percentile(samples, 50)),
                        p95_ms=float(np.percentile(samples, 95)),
                        p99_ms=float(np.percentile(samples, 99)),
                        max_abs_point_difference=differences[mode],
                    )
                )
    report.update(
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        sources_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "src/mars_titan/memory/financial_session.py",
                ROOT / "src/mars_titan/memory/financial_mechanisms.py",
                ROOT / "src/mars_titan/memory/native_mechanism_collection.py",
            )
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                cases=[{k: v for k, v in c.items() if k != "samples"} for c in report["cases"]],
                process_seconds=report["process_seconds"],
            )
        )
    )


if __name__ == "__main__":
    main()
