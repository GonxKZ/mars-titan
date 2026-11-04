"""Coste del lector E2 con fixtures. La preparación MAC queda fuera del tiempo por llamada."""

import hashlib
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import torch

import mars_titan
from mars_titan.memory.spaced_utility_readout import apply_spaced_utility_readout
from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
from mars_titan.models.titans.financial import DecisionBatch, FinancialConfig, FinancialPredictor

# Los datos se generan con el mismo contrato mínimo de las pruebas, sin abrir un corpus.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests/models/titans"))
from test_financial_adapter import raw_batch, specification  # noqa: E402

AT, CODEC, CONTEXT = 1_609_459_200_000_000, "a" * 64, "b" * 64


def fixture(count):
    root = Path(__file__).resolve().parents[3]
    if not Path(mars_titan.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError(
            "El benchmark necesita las fuentes de este checkout con PYTHONPATH=src:."
        )
    spec = specification()
    core = FinancialPredictor(
        FinancialConfig(spec, variant="mac_online", hidden_size=32, layers=1, max_batch=count),
        dtype=torch.float32,
    ).eval()
    core.requires_grad_(False)
    raw = raw_batch(at=AT)
    raw["inputs"] = {
        name: np.repeat(value[:1], count, axis=0) for name, value in raw["inputs"].items()
    }
    raw["presence"] = np.repeat(raw["presence"][:1], count, axis=0)
    raw["sample_ids"] = [f"US/FIXTURE_{i}/{AT}" for i in range(count)]
    raw["market"] = ["US"] * count
    raw["prediction_at"] = np.full(count, AT, dtype="datetime64[us]")
    raw["input_available_at"] = np.full(count, AT - 1, dtype="datetime64[us]")
    batch = DecisionBatch.from_corpus(raw, spec, dtype=torch.float32)
    prepared = core.prepare(batch, core.initial_state(batch.flow_ids))
    model = EpisodicReadout(
        EpisodicReadoutConfig(
            CODEC, hidden_size=32, neighbors=8, max_batch=count, max_working_bytes=256 * 1024**2
        ),
        dtype=torch.float32,
    ).eval()
    model.requires_grad_(False)
    generator = torch.Generator().manual_seed(29)
    keys = torch.randn((count, 64), generator=generator, dtype=torch.float64)
    keys = (keys / torch.linalg.vector_norm(keys, dim=1, keepdim=True)).float()
    data = EpisodeSnapshot.create(
        keys=keys,
        values=torch.randn((count, 64), generator=generator) / 10,
        labels=torch.arange(count, dtype=torch.float64) / count / 100,
        ids=torch.arange(1, count + 1, dtype=torch.int64),
        decision_at=torch.full((count,), AT - 2, dtype=torch.int64),
        available_at=torch.full((count,), AT - 3, dtype=torch.int64),
        maturity_at=torch.full((count,), AT - 1, dtype=torch.int64),
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=AT,
        dtype=torch.float32,
    )
    return core, prepared, model, data


def sample(core, prepared, model, snapshot, enabled):
    before = time.perf_counter_ns()
    prediction, trace = apply_spaced_utility_readout(
        prepared, core.head, model, snapshot, enabled=enabled, context_id=CONTEXT, cutoff=AT
    )
    times = dict.fromkeys(prepared.next_state.flow_ids, AT + 1)
    if enabled:
        rows = trace.ledger_rows(times)
    else:
        rows = [
            dict(flow_id=flow, point=point, deleted_points=[], label_not_before=AT + 1)
            for flow, point in zip(
                prepared.next_state.flow_ids,
                prediction.point_predictions.cpu().tolist(),
                strict=True,
            )
        ]
    payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
    return (time.perf_counter_ns() - before) / 1e6, rows, len(payload.encode())


def main():
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    rng = torch.get_rng_state().clone()
    started, measurements = time.perf_counter(), []
    for count in (64, 256, 1024):
        core, prepared, model, data = fixture(count)
        samples = {False: [], True: []}
        for repetition in range(12):
            points = []
            for enabled in (False, True) if repetition % 2 == 0 else (True, False):
                elapsed, rows, payload_bytes = sample(core, prepared, model, data, enabled)
                points.append([r["point"] for r in rows])
                if enabled:
                    diagnostic_bytes = payload_bytes
                if repetition >= 2:
                    samples[enabled].append(elapsed)
            assert points[0] == points[1]
        tracemalloc.start()
        sample(core, prepared, model, data, True)
        _, python_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        data.verify()
        measurements.append(
            dict(
                rows=count,
                snapshot_episodes=count,
                neighbors=8,
                hidden_size=32,
                dtype="torch.float32",
                warmup_per_mode=2,
                repetitions_per_mode=10,
                original_ms=samples[False],
                diagnostic_ms=samples[True],
                original_median_ms=statistics.median(samples[False]),
                diagnostic_median_ms=statistics.median(samples[True]),
                diagnostic_json_bytes=diagnostic_bytes,
                python_allocation_peak_bytes=python_peak,
                snapshot_tensor_bytes=sum(data.usage().values()),
                working_budget_bytes=model.config.max_working_bytes,
                estimated_working_bytes=model.estimated_bytes(count, count, refinements=9),
                emitted_point_parity=True,
                mac_memory_steps=sorted(set(prepared.next_state.mac.memory.steps.tolist())),
            )
        )
    assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
    report = dict(
        kind="technical_readout_fixture_cost",
        torch=str(torch.__version__),
        python=platform.python_version(),
        cpu_threads=2,
        device="cpu",
        cuda_initialized=False,
        optimizer_steps=0,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        process_seconds_without_imports=time.perf_counter() - started,
        process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        measurements=measurements,
        limits=[
            "Fixtures compactos y padre sin entrenamiento, no formas completas del corpus.",
            "Incluye lector, cabeza, supresiones, conversión de filas y JSON, sin disco.",
            "MAC se prepara una vez por tamaño, fuera del cronómetro por llamada.",
            "No incluye consolidación, ledger, persistencia durable, GPU ni transferencias.",
            "Tracemalloc se mide en otra pasada y excluye las asignaciones nativas.",
            "RSS es el pico absoluto del proceso completo, no el incremento de E2.",
            "Diez muestras por modo no ofrecen una estimación precisa de p95 o p99.",
            "No se midieron energía ni error predictivo sobre datos financieros.",
        ],
    )
    Path(sys.argv[1]).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            [
                dict(
                    rows=m["rows"],
                    original=m["original_median_ms"],
                    diagnostics=m["diagnostic_median_ms"],
                )
                for m in measurements
            ]
        )
    )


if __name__ == "__main__":
    main()
