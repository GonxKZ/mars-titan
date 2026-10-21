"""Coste nativo de emitir, guardar trazas y madurar fixtures, sin aprendizaje externo."""

import argparse
import hashlib
import json
import os
import platform
import resource
import runpy
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "reports/engineering/mechanism-window-20261021/benchmark.py"
helper = runpy.run_path(str(HELPER))

from mars_titan.memory.financial_session import FinancialPhase, FinancialSession  # noqa: E402
from mars_titan.memory.retention_bank import RetentionConfig  # noqa: E402
from mars_titan.models.titans.episodic_readout import (  # noqa: E402
    EpisodicReadout,
    EpisodicReadoutConfig,
)
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig  # noqa: E402
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor  # noqa: E402
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer  # noqa: E402


def stats(samples):
    return dict(
        samples_ms=samples,
        p50_ms=float(np.percentile(samples, 50)),
        p95_ms=float(np.percentile(samples, 95)),
        p99_ms=float(np.percentile(samples, 99)),
        calls_per_second=1000 / float(np.mean(samples)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", choices=(4, 64, 256), default=(4, 64, 256))
    parser.add_argument("--refinements", type=int, nargs="+", choices=(1, 4), default=(1, 4))
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
        or not args.storage_root.is_dir()
    ):
        raise ValueError("Se necesita una salida nueva y almacenamiento externo existente")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0, sin alternativa CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = helper["load_native"](os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    moment = helper["moment"]
    report = dict(
        status="running_not_complete",
        requested_rows=list(args.rows),
        requested_refinements=list(args.refinements),
        recorded_at_utc=datetime.now(UTC).isoformat(),
        scope="native_fixture_decision_and_following_maturation",
        code_parent_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        cpu_threads=torch.get_num_threads(),
        device="cuda:0",
        dtype="torch.float64",
        autocast=False,
        tf32=False,
        warmup=2,
        repetitions=5,
        real_corpus=False,
        outer_optimizer_steps=0,
        predictive_improvement_measured=False,
        native_sha256=native.binary_sha256,
        energy_measured=False,
        cases=[],
    )

    def save():
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        )

    save()
    started = time.perf_counter()
    for rows in args.rows:
        source = helper["flow_source"](
            args.storage_root / f"source-{rows}", flows=tuple(f"US/T{i:04d}" for i in range(rows))
        )
        model = (
            FinancialPredictor(
                FinancialConfig(source["spec"], variant="mac_online", hidden_size=32),
                device="cuda:0",
                dtype=torch.float64,
            )
            .eval()
            .requires_grad_(False)
        )
        parent_id = model._parameter_id
        bank_hash = None
        for k in args.refinements:
            for mode in ("legacy", "bridge"):
                common = dict(codec_id=source["codec"].fingerprint(), hidden_size=32, neighbors=8)
                bridge = EvidenceBridgeConfig(**common, cycles=2) if mode == "bridge" else None
                reader = (
                    EpisodicReadout(
                        EpisodicReadoutConfig(**common, refinements=k, evidence_bridge=bridge),
                        device="cuda:0",
                        dtype=torch.float64,
                    )
                    .eval()
                    .requires_grad_(False)
                )
                consumer = FrozenFinancialConsumer(model, readout=reader)
                decision, maturity, peaks, allocations = [], [], [], []
                trace_bytes, observed_work = 0, None
                for repetition in range(7):
                    path = args.storage_root / f"session-{rows}-{k}-{mode}-{repetition}"
                    with FinancialSession(
                        path,
                        native=native,
                        consumer=consumer,
                        codec=source["codec"],
                        prefixes=source["prefixes"],
                        retention=RetentionConfig(capacity=256, frontier=8, new_candidates=8),
                        phase=FinancialPhase(
                            "validation", moment(125), moment(126), moment(130), moment(131)
                        ),
                        admission="m1",
                        block_rows=64,
                        world="bridge_cost_fixture",
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
                        assert bank_hash == probe._digest
                        del probe, bank
                        torch.cuda.synchronize(0)
                        before = torch.cuda.memory_allocated(0)
                        torch.cuda.reset_peak_memory_stats(0)
                        tick = time.perf_counter_ns()
                        second = run.step(source["batches"][128], [])
                        torch.cuda.synchronize(0)
                        decision_ms = (time.perf_counter_ns() - tick) / 1e6
                        peak = torch.cuda.max_memory_allocated(0)
                        reference = run._bundle(run.snapshot()["state"])["control"]
                        control = run._read(reference, "control")
                        assert control["mac_updates"] == rows and control["refinements"] == k
                        assert len(second.predictions) == rows
                        trace_bytes = reference["bytes"]
                        observed_work = {
                            key: value
                            for key, value in control.items()
                            if key.startswith("bridge_")
                        }
                        labels = [
                            native.Feedback(p.id, 0, moment(129), 0.125) for p in second.predictions
                        ]
                        torch.cuda.synchronize(0)
                        tick = time.perf_counter_ns()
                        run.step(
                            [], labels, kind="settlement", cutoff=moment(131), close_phase=True
                        )
                        torch.cuda.synchronize(0)
                        maturity_ms = (time.perf_counter_ns() - tick) / 1e6
                        assert run.snapshot()["applied"] == 2 * rows
                        if repetition >= 2:
                            decision.append(decision_ms)
                            maturity.append(maturity_ms)
                            peaks.append(peak)
                            allocations.append(before)
                    del run
                report["cases"].append(
                    dict(
                        rows=rows,
                        refinements=k,
                        mode=mode,
                        logical_assets="replicated_price_fixture_not_independent_histories",
                        input_spec_id=source["spec"].fingerprint(),
                        parent_parameter_id=parent_id,
                        eligible_bank_sha256=bank_hash,
                        reader=reader.get_extra_state(),
                        reader_parameter_count=sum(p.numel() for p in reader.parameters()),
                        native_block_rows=64,
                        decision=stats(decision),
                        following_maturation=stats(maturity),
                        serial_sum=stats([a + b for a, b in zip(decision, maturity, strict=True)]),
                        decision_rows_per_second=rows * 1000 / float(np.mean(decision)),
                        allocated_before_bytes=max(allocations),
                        allocated_peak_bytes=max(peaks),
                        control_archive_bytes=trace_bytes,
                        observed_work=observed_work,
                    )
                )
                del reader, consumer
                torch.cuda.empty_cache()
                save()
        del source, model
        torch.cuda.empty_cache()
    paths = [
        Path(__file__),
        HELPER,
        ROOT / "tests/memory/test_financial_session_controls.py",
        ROOT / "uv.lock",
    ]
    paths.extend(
        ROOT / ("src/mars_titan/models/titans/" + name + ".py")
        for name in (
            "episodic_readout",
            "evidence_bridge",
            "regional_episode_readout",
            "frozen_financial",
        )
    )
    paths.extend(
        ROOT / ("src/mars_titan/memory/" + name + ".py")
        for name in ("financial_consumers", "financial_session", "regional_episode_trace")
    )
    report.update(
        status="complete",
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    )
    save()


if __name__ == "__main__":
    main()
