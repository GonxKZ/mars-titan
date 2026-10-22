"""Coste de rutas fijas del controlador con fixtures, sin entrenamientos."""

import argparse
import hashlib
import json
import os
import resource
import runpy
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
from mars_titan.memory.retention_bank import RetentionConfig
from mars_titan.models.quantile_head import QUANTILE_HEAD
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "reports/engineering/mechanism-window-20261021/benchmark.py"


def statistics(samples):
    return dict(
        samples_ms=samples,
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=float(np.percentile(samples, 99)),
        mean_ms=float(np.mean(samples)),
        standard_deviation_ms=float(np.std(samples)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", choices=(4, 64, 256), default=(4, 64, 256))
    parser.add_argument("--routes", nargs="+", choices=("k1", "k4cr"), default=("k1", "k4cr"))
    parser.add_argument("--reader-max-batch", type=int, choices=(64, 256), default=256)
    args = parser.parse_args()
    if not args.storage_root.is_dir() or not args.output.parent.is_dir():
        raise ValueError("Se necesita almacenamiento externo existente")
    if not torch.cuda.is_available():
        raise RuntimeError("La medida requiere cuda:0, sin alternativa CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    helper = runpy.run_path(str(HELPER))
    native = helper["load_native"](os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    moment = helper["moment"]
    report = dict(
        status="running_not_complete",
        requested_rows=list(args.rows),
        requested_routes=list(args.routes),
        reader_max_batch=args.reader_max_batch,
        recorded_at_utc=datetime.now(UTC).isoformat(),
        parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        threads=torch.get_num_threads(),
        device="cuda:0",
        dtype="float64",
        tf32=False,
        autocast=False,
        warmup=1,
        repetitions=3,
        real_corpus=False,
        optimizer_steps=0,
        scope="native_fixed_mechanism_decision_and_following_settlement",
        cases=[],
        native_sha256=native.binary_sha256,
        source_sha256={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                HELPER,
                ROOT / "uv.lock",
                *(
                    ROOT / "src/mars_titan/models/titans" / (name + ".py")
                    for name in (
                        "mechanism_consumer",
                        "mechanism_catalog",
                        "regional_routes",
                        "regional_mechanism_trace",
                        "mechanism_policy",
                    )
                ),
                ROOT / "src/mars_titan/memory/financial_mechanisms.py",
            )
        },
    )
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    started = time.perf_counter()
    for rows in args.rows:
        source = helper["flow_source"](
            args.storage_root / f"source-{rows}", flows=tuple(f"US/T{i:04d}" for i in range(rows))
        )
        parent = (
            FinancialPredictor(
                FinancialConfig(
                    source["spec"], variant="mac_online", hidden_size=32, head=QUANTILE_HEAD
                ),
                device="cuda:0",
                dtype=torch.float64,
            )
            .eval()
            .requires_grad_(False)
        )
        bank_hash = None
        for route in args.routes:
            for mode in ("legacy", "regional"):
                common = dict(
                    codec_id=source["codec"].fingerprint(),
                    hidden_size=32,
                    max_batch=args.reader_max_batch,
                )
                bridge = EvidenceBridgeConfig(**common, cycles=2) if mode == "regional" else None
                reader = (
                    EpisodicReadout(
                        EpisodicReadoutConfig(**common, evidence_bridge=bridge),
                        device="cuda:0",
                        dtype=torch.float64,
                    )
                    .eval()
                    .requires_grad_(False)
                )
                consumer = FrozenMechanismConsumer(
                    parent,
                    readout=reader,
                    action=MechanismAction(route=route),
                    catalog=MechanismCatalog(
                        reader_contract="shared_evidence_bridge_v1" if bridge else "legacy"
                    ),
                )
                decisions, settlements, peaks = [], [], []
                archive_bytes, work = None, None
                for repetition in range(4):
                    with FinancialSession(
                        args.storage_root / f"session-{rows}-{route}-{mode}-{repetition}",
                        native=native,
                        consumer=consumer,
                        codec=source["codec"],
                        prefixes=source["prefixes"],
                        retention=RetentionConfig(capacity=256),
                        phase=FinancialPhase(
                            "validation", moment(125), moment(126), moment(130), moment(131)
                        ),
                        admission="m1",
                        block_rows=64,
                        world="regional_controller_cost_fixture",
                        fold="0",
                    ) as run:
                        run.step(source["batches"][125], [], kind="warmup")
                        first = run.step(source["batches"][126], [])
                        labels = [
                            native.Feedback(p.id, 0, moment(127), 0.125) for p in first.predictions
                        ]
                        run.step([], labels, kind="settlement", cutoff=moment(127))
                        bank, _ = run._bank(run._bundle(run.snapshot()["state"])["bank"])
                        probe = run._binding.snapshot(bank, "e" * 64, moment(128))
                        probe.verify()
                        assert probe.count == rows
                        bank_hash = probe._digest if bank_hash is None else bank_hash
                        assert probe._digest == bank_hash
                        del probe, bank
                        torch.cuda.synchronize(0)
                        torch.cuda.reset_peak_memory_stats(0)
                        tick = time.perf_counter_ns()
                        second = run.step(source["batches"][128], [])
                        torch.cuda.synchronize(0)
                        decision_ms = (time.perf_counter_ns() - tick) / 1e6
                        peak = torch.cuda.max_memory_allocated(0)
                        control_ref = run._bundle(run.snapshot()["state"])["control"]
                        control = run._read(control_ref, "control")
                        work = control["work"]
                        archive_bytes = control_ref["bytes"]
                        assert work["mac_online_observations"] == rows
                        assert work["reader_refinements"] == rows * int(route[1])
                        labels = [
                            native.Feedback(p.id, 0, moment(129), 0.125) for p in second.predictions
                        ]
                        torch.cuda.synchronize(0)
                        tick = time.perf_counter_ns()
                        run.step(
                            [], labels, kind="settlement", cutoff=moment(131), close_phase=True
                        )
                        torch.cuda.synchronize(0)
                        settlement_ms = (time.perf_counter_ns() - tick) / 1e6
                        assert run.snapshot()["applied"] == rows * 2
                        if repetition:
                            decisions.append(decision_ms)
                            settlements.append(settlement_ms)
                            peaks.append(peak)
                report["cases"].append(
                    dict(
                        rows=rows,
                        route=route,
                        mode=mode,
                        parent_parameter_id=parent._parameter_id,
                        bank_sha256=bank_hash,
                        inputs_id=source["spec"].fingerprint(),
                        consumer_identity=consumer.identity(),
                        decision=statistics(decisions),
                        following_settlement=statistics(settlements),
                        control_bytes=archive_bytes,
                        allocated_peak_bytes=max(peaks),
                        decision_rows_per_second=rows * 1000 / float(np.mean(decisions)),
                        work=work,
                    )
                )
                del reader, consumer, run
                torch.cuda.empty_cache()
                save()
    report.update(
        status="complete",
        elapsed_seconds=time.perf_counter() - started,
        maximum_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )
    save()


if __name__ == "__main__":
    main()
