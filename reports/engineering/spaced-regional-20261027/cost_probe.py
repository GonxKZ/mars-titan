"""Coste del productor regional E2 con fixtures compactos y sin entrenamiento externo."""

import hashlib
import importlib.util
import json
import platform
import resource
import statistics
import sys
import time
import tracemalloc
from dataclasses import replace
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import mars_titan
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory import spaced_regional_readout
from mars_titan.models.titans.bridge_hierarchy import BridgeHierarchyConfig
from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig, BridgeStoppingConfig
from mars_titan.models.titans.episodic_readout import EpisodicReadout
from mars_titan.models.titans.evidence_bridge import EvidenceBridgeConfig

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / "reports/engineering/spaced-readout-20261027/cost_probe.py"


def fixture(count, kind, prepare):
    core, prepared, original, data = prepare(count)
    values = dict(
        codec_id=original.config.codec_id,
        hidden_size=original.config.hidden_size,
        neighbors=original.config.neighbors,
        max_batch=count,
        max_working_bytes=original.config.max_working_bytes,
    )
    if kind == "j4":
        values.update(cycles=0, recurrence=BridgeRecurrenceConfig(4, 1216))
    elif kind == "adaptive_compact":
        values.update(
            cycles=0,
            recurrence=BridgeRecurrenceConfig(
                4, 1216, stopping=BridgeStoppingConfig(0.01, 0.0001, backend=kind)
            ),
        )
    elif kind == "tree":
        values.update(
            cycles=0,
            topology="self",
            regions=("one", "two", "three", "four"),
            hierarchy=BridgeHierarchyConfig(),
        )
    config = replace(original.config, evidence_bridge=EvidenceBridgeConfig(**values))
    model = EpisodicReadout(config, dtype=torch.float32).eval()
    model.requires_grad_(False)
    return core, prepared, model, data


def sample(core, prepared, model, data, enabled):
    started = time.perf_counter_ns()
    prediction, trace = spaced_regional_readout.apply_regional_spaced_utility_readout(
        prepared,
        core.head,
        model,
        data,
        enabled=enabled,
        context_id=data.context_id,
        cutoff=data.cutoff,
    )
    if trace is None:
        rows = [
            dict(flow_id=flow, point=value, deleted_points=[], label_not_before=data.cutoff + 1)
            for flow, value in zip(
                prepared.next_state.flow_ids,
                prediction.point_predictions.cpu().tolist(),
                strict=True,
            )
        ]
    else:
        rows = trace.ledger_rows(dict.fromkeys(prepared.next_state.flow_ids, data.cutoff + 1))
    payload = json.dumps(rows, separators=(",", ":"), allow_nan=False)
    elapsed = (time.perf_counter_ns() - started) / 1e6
    return elapsed, [row["point"] for row in rows], len(payload.encode())


def main():
    if not Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("El benchmark necesita las fuentes propias con PYTHONPATH=src:.")
    target = Path(sys.argv[1])
    if target.exists():
        raise ValueError("El recibo de coste necesita un destino nuevo")
    spec = importlib.util.spec_from_file_location("regional_fixture_helper", HELPER)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    sys.path.insert(0, str(ROOT / "tests/memory"))
    from test_financial_session_controls import no_target_estimation

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    rng, measurements = torch.get_rng_state().clone(), []
    attempts = 0

    def forbidden(*_):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("La medida técnica no admite pasos de optimizador")

    hook = register_optimizer_step_pre_hook(forbidden)
    guard = no_target_estimation.__wrapped__()
    next(guard)
    started = time.perf_counter()
    try:
        for rows in (64, 256):
            for kind in ("ring", "j4", "adaptive_compact", "tree"):
                core, prepared, model, data = fixture(rows, kind, helper.fixture)
                values = {False: [], True: []}
                for repetition in range(12):
                    points = []
                    for enabled in (False, True) if repetition % 2 == 0 else (True, False):
                        elapsed, emitted, size = sample(core, prepared, model, data, enabled)
                        points.append(emitted)
                        if repetition >= 2:
                            values[enabled].append(elapsed)
                        if enabled:
                            diagnostic_bytes = size
                    assert points[0] == points[1]
                tracemalloc.start()
                sample(core, prepared, model, data, True)
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                data.verify()
                measurements.append(
                    dict(
                        rows=rows,
                        kind=kind,
                        regions=len(model.config.evidence_bridge.regions),
                        snapshot_episodes=data.count,
                        neighbors=model.config.neighbors,
                        original_ms=values[False],
                        diagnostic_ms=values[True],
                        original_median_ms=statistics.median(values[False]),
                        diagnostic_median_ms=statistics.median(values[True]),
                        diagnostic_json_bytes=diagnostic_bytes,
                        python_allocation_peak_bytes=peak,
                        snapshot_tensor_bytes=sum(data.usage().values()),
                        estimated_working_bytes=model.estimated_bytes(
                            rows, data.count, refinements=9
                        ),
                        working_budget_bytes=model.config.max_working_bytes,
                        emitted_point_parity=True,
                        memory_steps=sorted(set(prepared.next_state.mac.memory.steps.tolist())),
                    )
                )
        assert torch.equal(rng, torch.get_rng_state()) and not torch.cuda.is_initialized()
        result = dict(
            kind="technical_regional_spaced_readout_cost",
            hardware=dict(cpu=cpu_name(), platform=platform.platform()),
            python=platform.python_version(),
            torch=str(torch.__version__),
            device="cpu",
            dtype="torch.float32",
            cpu_threads=2,
            hidden_size=32,
            warmup_per_mode=2,
            repetitions_per_mode=10,
            measurements=measurements,
            optimizer_attempts=attempts,
            cuda_initialized=False,
            seconds_without_imports=time.perf_counter() - started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            producer_sha256=hashlib.sha256(
                Path(spaced_regional_readout.__file__).read_bytes()
            ).hexdigest(),
            fixture_helper_sha256=hashlib.sha256(HELPER.read_bytes()).hexdigest(),
            probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            limits=[
                "Fixtures compactos de anchuras 5/2/2/3/3, no formas del corpus completo.",
                "Incluye lector, cabeza, supresiones, filas y JSON. "
                "MAC se prepara fuera de cada llamada.",
                "El productor conserva la lectura original y añade "
                "una proyección del pool para los diagnósticos.",
                "Excluye ledger, consolidación, persistencia durable, "
                "disco, CUDA y transferencias GPU.",
                "Tracemalloc se mide en otra pasada, sin asignaciones nativas. RSS es absoluto.",
                "No se miden energía, generalización financiera ni p95/p99 fiables.",
            ],
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(
            json.dumps(
                [
                    dict(
                        rows=v["rows"],
                        kind=v["kind"],
                        original=v["original_median_ms"],
                        e2=v["diagnostic_median_ms"],
                    )
                    for v in measurements
                ]
            )
        )
    finally:
        guard.close()
        hook.remove()


if __name__ == "__main__":
    main()
