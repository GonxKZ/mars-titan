"""Coste de bancos históricos y preparación financiera con fixtures, sin ajuste externo."""

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

from test_financial_session_controls import flow_source  # noqa: E402

from mars_titan.models.titans.bank_readout import BankReadoutConfig  # noqa: E402
from mars_titan.models.titans.config import MemoryConfig, MemoryMultiscale  # noqa: E402
from mars_titan.models.titans.financial_inputs import DecisionBatch  # noqa: E402
from mars_titan.models.titans.neural_memory import NeuralMemory  # noqa: E402
from mars_titan.training.titans_walk_forward import _predictor  # noqa: E402


def summarize(values):
    return dict(
        samples_ms=values,
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        calls_per_second=1000 / float(np.mean(values)),
    )


def raw(blocks):
    return dict(
        inputs={k: np.concatenate([b.inputs[k] for b in blocks]) for k in blocks[0].inputs},
        presence=np.concatenate([b.presence for b in blocks]),
        sample_ids=[v for b in blocks for v in b.sample_ids],
        prediction_at=np.array(
            [v for b in blocks for v in b.prediction_at], dtype="datetime64[us]"
        ),
        input_available_at=np.array(
            [v for b in blocks for v in b.input_available_at], dtype="datetime64[us]"
        ),
    )


def profile_read(model, query, state):
    try:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        ) as profile:
            for _ in range(3):
                model.read(query, state)
            torch.cuda.synchronize(0)
        events = sorted(profile.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)
        return [
            dict(
                name=e.key,
                calls=e.count,
                self_cpu_us=e.self_cpu_time_total,
                self_device_us=getattr(e, "self_device_time_total", None),
            )
            for e in events[:20]
        ]
    except RuntimeError as error:
        return dict(unavailable=str(error))


def prepare_financial(model, states, inputs):
    proposals = []
    for state, values in zip(states, inputs, strict=True):
        batch = DecisionBatch.from_corpus(
            values, model.config.inputs, device="cuda:0", dtype=torch.float32
        )
        proposals.append(model.prepare(batch, state))
    return torch.cat([item.quantiles for item in proposals]), proposals


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink() or not args.output.parent.is_dir():
        raise ValueError("La salida debe ser una ruta nueva dentro de un directorio existente")
    if not torch.cuda.is_available():
        raise RuntimeError("Se necesita cuda:0, sin alternativa CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    declaration = json.loads((ROOT / "configs/experiments/bank-readout-v1.json").read_text())
    recipe = json.loads((ROOT / declaration["base_recipe"]).read_text())
    result = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        torch_git=torch.version.git_version,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        device="cuda:0",
        cpu_threads=torch.get_num_threads(),
        warmup=3,
        repetitions=20,
        real_corpus=False,
        optimizer_steps=0,
        cases=[],
        profiles=[],
        status="running_not_complete",
    )
    start = time.perf_counter()
    for dtype in (torch.float32, torch.float64):
        for count in (1, 64, 256):
            for route in (None, "uniform", "constant", "query"):
                cfg = MemoryConfig(
                    dim=64,
                    depth=2,
                    max_batch=256,
                    max_tokens=8,
                    max_state_bytes=128 * 1024**2,
                    multiscale=MemoryMultiscale(form="ema_bank"),
                    bank_readout=BankReadoutConfig(routing=route) if route else None,
                )
                model = NeuralMemory(cfg, device="cuda:0", dtype=dtype).eval().requires_grad_(False)
                x = torch.linspace(-0.2, 0.3, count * 64, device="cuda:0", dtype=dtype).reshape(
                    count, 1, 64
                )
                state = model.update(x, model.initial_state(count))
                for _ in range(3):
                    model.read(x, state)
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                resident = torch.cuda.memory_allocated(0)
                values = []
                for _ in range(20):
                    torch.cuda.synchronize(0)
                    tick = time.perf_counter_ns()
                    read = model.read(x, state)
                    torch.cuda.synchronize(0)
                    values.append((time.perf_counter_ns() - tick) / 1e6)
                result["cases"].append(
                    dict(
                        scope="memory_read",
                        rows=count,
                        dtype=str(dtype),
                        route=route or "mean_weights",
                        configuration=cfg.identity(),
                        **summarize(values),
                        allocated_resident_bytes=resident,
                        allocated_peak_bytes=torch.cuda.max_memory_allocated(0),
                        reserved_peak_bytes=torch.cuda.max_memory_reserved(0),
                        finite=bool(torch.isfinite(read).all()),
                        output_sha256=hashlib.sha256(
                            read.detach().cpu().numpy().tobytes()
                        ).hexdigest(),
                    )
                )
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                if dtype == torch.float32 and count in (1, 256) and route in (None, "query"):
                    result["profiles"].append(
                        dict(
                            rows=count,
                            route=route or "mean_weights",
                            events=profile_read(model, x, state),
                        )
                    )
                del read, state, model, x
                torch.cuda.empty_cache()
    with tempfile.TemporaryDirectory(prefix="mars-bank-readout-cost-") as folder:
        for count in (1, 64, 256):
            source = flow_source(
                Path(folder) / f"source-{count}", flows=tuple(f"US/T{i:04d}" for i in range(count))
            )
            block_rows = recipe["recipe"]["block_rows"]
            inputs = {
                index: [
                    raw(source["batches"][index][i : i + block_rows])
                    for i in range(0, count, block_rows)
                ]
                for index in (125, 126, 127, 128)
            }
            for arm in declaration["arms"]:
                model, _ = _predictor(
                    recipe,
                    source["spec"],
                    "mac_online",
                    42,
                    "cuda:0",
                    changes=declaration["arms"][arm]["predictor_changes"],
                )
                model.eval().requires_grad_(False)
                states = []
                for block in range(len(inputs[128])):
                    state = None
                    for index in (125, 126, 127):
                        batch = DecisionBatch.from_corpus(
                            inputs[index][block],
                            model.config.inputs,
                            device="cuda:0",
                            dtype=torch.float32,
                        )
                        state = model.initial_state(batch.flow_ids) if state is None else state
                        state = model.prepare(batch, state).next_state
                    states.append(state)

                for _ in range(3):
                    prepare_financial(model, states, inputs[128])
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                resident = torch.cuda.memory_allocated(0)
                values = []
                for _ in range(20):
                    torch.cuda.synchronize(0)
                    tick = time.perf_counter_ns()
                    quantiles, proposals = prepare_financial(model, states, inputs[128])
                    torch.cuda.synchronize(0)
                    values.append((time.perf_counter_ns() - tick) / 1e6)
                result["cases"].append(
                    dict(
                        scope="financial_prepare_with_input_validation_and_transfer",
                        rows=count,
                        arm=arm,
                        dtype="torch.float32",
                        **summarize(values),
                        configuration=model.config.identity(),
                        parameter_id=model._parameter_id,
                        input_spec_id=source["spec"].fingerprint(),
                        allocated_resident_bytes=resident,
                        allocated_peak_bytes=torch.cuda.max_memory_allocated(0),
                        reserved_peak_bytes=torch.cuda.max_memory_reserved(0),
                        finite=bool(torch.isfinite(quantiles).all()),
                        counts=model.operation_counts(count),
                        quantiles=quantiles.detach().cpu().tolist(),
                        physical_block_rows=block_rows,
                        blocks=len(states),
                    )
                )
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
                print(
                    json.dumps(dict(rows=count, arm=arm, p50_ms=float(np.median(values)))),
                    flush=True,
                )
                del proposals, quantiles, states, state, model, batch
                torch.cuda.empty_cache()
    result.update(
        status="complete",
        process_seconds=time.perf_counter() - start,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_sha256={
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                ROOT / "src/mars_titan/models/titans/neural_memory.py",
                ROOT / "src/mars_titan/models/titans/bank_readout.py",
                ROOT / "src/mars_titan/models/titans/financial.py",
                ROOT / "configs/experiments/bank-readout-v1.json",
            )
        },
    )
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
