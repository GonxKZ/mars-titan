"""Propuesta, recuperación y continuación E2 con registros nativos de fixtures."""

import hashlib
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import torch

import mars_titan
from mars_titan.memory.financial_consumers import bank_snapshot
from mars_titan.memory.native_backend import load_native
from mars_titan.memory.spaced_retention import POLICIES, SpacedRetentionConfig, SpacedUtilityBank
from mars_titan.memory.spaced_utility import DAY_US, SpacedUtilityLedger

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/memory"))
from test_retention_bank import record  # noqa: E402

CODEC = "a" * 64


def fixture(native, count, policy):
    if not Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("El benchmark necesita PYTHONPATH=src:. y las fuentes de este checkout")
    memory = SpacedUtilityBank(
        native,
        SpacedRetentionConfig(policy=policy, capacity=count, frontier=8),
        codec_id=CODEC,
        model_id="d" * 64,
        world="fixture",
        partition="train",
        fold="0",
    )
    ledger = SpacedUtilityLedger(CODEC, memory.evidence_config, scope_id=memory.fingerprint())
    ledger.begin(2 * count + 2)
    ledger.publish()
    first = memory.propose(
        [record(native, i) for i in range(1, count + 1)], confirmed_at=ledger.clock, evidence=ledger
    )
    memory, ledger = first.bank, first.evidence
    for at in (2 * count + 10, DAY_US + 2 * count + 10):
        snapshot = bank_snapshot(
            memory,
            codec_id=CODEC,
            context_id="b" * 64,
            cutoff=at,
            dtype=torch.float64,
            device="cpu",
        )
        rows = [
            dict(
                flow_id=f"US/FIXTURE_{start}",
                point=0.0,
                deleted_points=[
                    (i, 1.0 + (i % 11) / 10) for i in range(start, min(start + 64, count + 1))
                ],
                label_not_before=at + 1,
            )
            for start in range(1, count + 1, 64)
        ]
        ledger.begin(at)
        ledger.issue(snapshot, rows)
        ledger.publish()
        ledger.begin(at + 1)
        ledger.receive([(r["flow_id"], at, at + 1, 0.0) for r in rows])
        ledger.publish()
    # La cola conserva una cohorte sin resultado, con referencias que pueden descartarse.
    at = ledger.clock + 1
    ledger.begin(at)
    snapshot = bank_snapshot(
        memory, codec_id=CODEC, context_id="b" * 64, cutoff=at, dtype=torch.float64, device="cpu"
    )
    for row in rows:
        row["label_not_before"] = None
    ledger.issue(snapshot, rows)
    ledger.publish()
    return memory, ledger


def cycle(native, memory, ledger):
    first_id = memory.seen + 1
    incoming = [record(native, i) for i in range(first_id, first_id + 64)]
    later = [record(native, i) for i in range(first_id + 64, first_id + 128)]
    before = time.perf_counter_ns()
    proposal = memory.propose(incoming, confirmed_at=ledger.clock, evidence=ledger)
    saved_bank, saved_ledger = proposal.bank.snapshot(), proposal.evidence.export()
    clone = memory.restore(saved_bank)
    restored = SpacedUtilityLedger.restore(
        CODEC, memory.evidence_config, saved_ledger, scope_id=memory.fingerprint()
    )
    clone.validate_evidence(restored)
    for value in (proposal.evidence, restored):
        value.begin(value.clock + 1)
        value.publish()
    following = proposal.bank.propose(
        later, confirmed_at=restored.clock, evidence=proposal.evidence
    )
    resumed = clone.propose(later, confirmed_at=restored.clock, evidence=restored)
    assert following.bank.snapshot() == resumed.bank.snapshot()
    assert following.evidence.export() == resumed.evidence.export()
    elapsed = (time.perf_counter_ns() - before) / 1e6
    metadata = {k: v for k, v in saved_bank.items() if k != "memory"}
    logical_bytes = len(saved_bank["memory"]) + len(json.dumps([metadata, saved_ledger]).encode())
    return elapsed, logical_bytes, proposal.bank.receipt["estimated_working_bytes"]


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    native = load_native()
    started, results = time.perf_counter(), []
    for count in (64, 256, 1024):
        for policy in POLICIES:
            memory, ledger = fixture(native, count, policy)
            samples = []
            for repetition in range(6):
                elapsed, size, estimate = cycle(native, memory, ledger)
                if repetition:
                    samples.append(elapsed)
            tracemalloc.start()
            cycle(native, memory, ledger)
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results.append(
                dict(
                    capacity=count,
                    policy=policy,
                    pending=ledger.usage(),
                    warmup=1,
                    repetitions=5,
                    cycle_ms=samples,
                    median_ms=statistics.median(samples),
                    logical_payload_bytes=size,
                    estimated_working_bytes=estimate,
                    python_peak_bytes=peak,
                    recovery_parity=True,
                )
            )
    report = dict(
        kind="technical_native_retention_cost",
        torch=str(torch.__version__),
        python=platform.python_version(),
        native_sha256=native.binary_sha256,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        device="cpu",
        cpu_threads=2,
        process_seconds_without_imports=time.perf_counter() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        results=results,
        limits=[
            "Incluye propuesta, snapshots, recuperación, conciliación "
            "y dos continuaciones emparejadas.",
            "Preparación de fixtures fuera del tiempo por ciclo. Sin predictor, disco ni GPU.",
            "Valores y pronósticos literales. No son datos financieros ni resultados predictivos.",
            "Pico Python en una pasada aparte. Excluye asignaciones nativas.",
            "Payload lógico sin framing del almacenamiento. RSS absoluto del proceso completo.",
            "Cinco muestras por caso no permiten estimar colas p95/p99 con precisión.",
            "Energía no medida.",
        ],
    )
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            [
                dict(capacity=r["capacity"], policy=r["policy"], median_ms=r["median_ms"])
                for r in results
            ]
        )
    )


if __name__ == "__main__":
    main()
