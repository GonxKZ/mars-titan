"""Coste del registro E2 con pronósticos literales. No ejecuta un predictor ni aprendizaje."""

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

from mars_titan.memory.spaced_utility import DAY_US, SpacedUtilityConfig, SpacedUtilityLedger
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

CODEC = "a" * 64


def fixture(count, at):
    keys = torch.zeros(count, 64, dtype=torch.float32)
    keys[:, 0] = 1
    snapshot = EpisodeSnapshot.create(
        keys=keys,
        values=torch.zeros_like(keys),
        labels=torch.zeros(count, dtype=torch.float64),
        ids=torch.arange(1, count + 1, dtype=torch.int64),
        decision_at=torch.ones(count, dtype=torch.int64),
        available_at=torch.zeros(count, dtype=torch.int64),
        maturity_at=torch.full((count,), 2, dtype=torch.int64),
        cutoff=at,
        codec_id=CODEC,
        context_id="b" * 64,
        dtype=torch.float64,
    )
    rows = [
        dict(
            flow_id=f"US/FIXTURE_{i}",
            point=0.0,
            deleted_points=[((i + j) % count + 1, 1.0) for j in range(8)],
            label_not_before=at + 1,
        )
        for i in range(count)
    ]
    return snapshot, rows


def cycle(count, fixtures, resume):
    run = SpacedUtilityLedger(CODEC, SpacedUtilityConfig(capacity=count))
    pending_bytes = 0
    started = time.perf_counter_ns()
    for at, (snapshot, rows) in zip((10, DAY_US + 10), fixtures, strict=True):
        run.begin(at)
        run.issue(snapshot, rows)
        assert run.publish() == []
        pending = json.dumps(run.export(), sort_keys=True, separators=(",", ":"))
        pending_bytes = max(pending_bytes, len(pending.encode()))
        if resume:
            run = SpacedUtilityLedger.restore(CODEC, run.config, json.loads(pending))
        run.begin(at + 1)
        run.receive([(r["flow_id"], at, at + 1, 0.0) for r in rows])
        assert len(run.publish()) == 1
    final = run.export()
    elapsed = (time.perf_counter_ns() - started) / 1e6
    assert all(s["qualified"] and s["mean_gain"] == 1 for s in run.scores().values())
    assert len(run.scores()) == count
    return elapsed, pending_bytes, final


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    rng = torch.get_rng_state().clone()
    started = time.perf_counter()
    measurements = []
    for count in (64, 256, 1024):
        fixtures = [fixture(count, at) for at in (10, DAY_US + 10)]
        _, _, reference = cycle(count, fixtures, False)
        samples = []
        for repetition in range(12):
            elapsed, size, final = cycle(count, fixtures, True)
            assert reference == final
            if repetition >= 2:
                samples.append(elapsed)
        tracemalloc.start()
        cycle(count, fixtures, True)
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        measurements.append(
            dict(
                rows_per_session=count,
                sessions_per_cycle=2,
                snapshot_episodes=count,
                deletions_per_row=8,
                warmup=2,
                repetitions=len(samples),
                cycle_ms=samples,
                median_ms=statistics.median(samples),
                rows_per_second=2000 * count / statistics.median(samples),
                max_pending_json_bytes=size,
                final_json_bytes=len(json.dumps(final, sort_keys=True).encode()),
                python_allocation_peak_bytes=python_peak,
                recovery_parity=True,
            )
        )
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    report = dict(
        kind="technical_fixture_bookkeeping_cost",
        python=platform.python_version(),
        torch=str(torch.__version__),
        platform=platform.platform(),
        cpu_threads=2,
        device="cpu",
        cuda_initialized=False,
        optimizer_steps=0,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        process_seconds_without_imports=time.perf_counter() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        measurements=measurements,
        limits=[
            "Incluye emisión, JSON, recuperación, resultados y publicación de dos sesiones.",
            "La preparación de fixtures queda fuera del tiempo por ciclo y dentro del proceso.",
            "No incluye predictor, producción de supresiones, disco, GPU ni transferencias.",
            "El pico tracemalloc excluye asignaciones nativas de PyTorch y se mide en otra pasada.",
            "RSS es el pico absoluto del proceso. No se midió el incremento atribuible a E2.",
            "Diez muestras no permiten estimar colas p95 o p99 con precisión.",
            "No hay observaciones financieras ni una mejora predictiva medida.",
        ],
    )
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            [dict(rows=m["rows_per_session"], median_ms=m["median_ms"]) for m in measurements]
        )
    )


if __name__ == "__main__":
    main()
