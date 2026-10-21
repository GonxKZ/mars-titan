"""Coste del extractor completo con fixtures, sin predictor ni aprendizaje nuevos."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]

from test_financial_adapter import raw_batch, specification  # noqa: E402
from test_mechanism_consumer import calibration  # noqa: E402
from test_mechanism_observations import observation_rows  # noqa: E402

from mars_titan.memory.mechanism_observations import extract_observation  # noqa: E402
from mars_titan.models.quantile_head import QUANTILE_HEAD  # noqa: E402
from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor  # noqa: E402
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer  # noqa: E402


def measure(call, *, warmup, repetitions, rows):
    samples, digests = [], []
    torch.cuda.reset_peak_memory_stats(0)
    for index in range(warmup + repetitions):
        torch.cuda.synchronize(0)
        start = time.perf_counter_ns()
        observation, tensor = call()
        torch.cuda.synchronize(0)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        if index >= warmup:
            samples.append(elapsed)
            digests.append(observation.sha256)
        if tensor is not None:
            expected = observation.tensor(device="cpu", dtype=tensor.dtype)
            if not torch.equal(tensor.cpu(), expected):
                raise RuntimeError("La transferencia cambia los rasgos calculados")
        del observation, tensor
    if len(set(digests)) != 1:
        raise RuntimeError("Las repeticiones no conservan la observación")
    return dict(
        samples_ms=samples,
        **{f"p{p}_ms": float(np.percentile(samples, p)) for p in (50, 95, 99)},
        mean_ms=float(np.mean(samples)),
        std_ms=float(np.std(samples, ddof=1)),
        rows_per_second=1000 * rows / float(np.mean(samples)),
        observation_sha256=digests[0],
        allocated_vram_peak_bytes=torch.cuda.max_memory_allocated(0),
        reserved_vram_peak_bytes=torch.cuda.max_memory_reserved(0),
    )


def extract_call(*, rows, engine, initial, flows, at, dtype, transfer):
    observed = extract_observation(
        rows, consumer=engine, state=initial, flow_ids=flows, cutoff=at, context_id="b" * 64
    )
    tensor = observed.tensor(device="cuda:0", dtype=dtype) if transfer else None
    return observed, tensor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=12)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
        or not 1 <= args.warmup <= 16
        or not 2 <= args.repetitions <= 128
        or len(set(args.rows)) != len(args.rows)
        or any(n not in (64, 256) for n in args.rows)
    ):
        raise ValueError("La medida necesita salida nueva, tamaños y repeticiones acotados")
    if not torch.cuda.is_available():
        raise RuntimeError("La medida necesita cuda:0 y no pasa a CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    start = time.perf_counter()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        device="cuda:0",
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        numpy=np.__version__,
        cpu_threads=torch.get_num_threads(),
        warmup=args.warmup,
        repetitions=args.repetitions,
        hidden_size=32,
        input_shapes=dict(prices=[64, 5], news=2, charts=2, fundamentals=3, macro=3),
        real_data_used=False,
        optimizer_steps=0,
        scientific_study=False,
        cases=[],
    )
    for dtype in (torch.float32, torch.float64):
        model = (
            FinancialPredictor(
                FinancialConfig(
                    specification(),
                    hidden_size=32,
                    max_batch=256,
                    head=QUANTILE_HEAD,
                    session_context="market_memory_v1",
                ),
                dtype=dtype,
                device="cuda:0",
            )
            .eval()
            .requires_grad_(False)
        )
        engine = FrozenMechanismConsumer(model, calibration=calibration())
        initial = engine.initial_state()
        for count in args.rows:
            for markets in (("US",), ("CN", "US")):
                raw = raw_batch()
                positions = np.arange(count) % 2
                raw["inputs"] = {k: v[positions] for k, v in raw["inputs"].items()}
                raw["presence"] = raw["presence"][positions]
                at = int(raw["prediction_at"][0].astype(np.int64))
                for key in ("prediction_at", "input_available_at"):
                    raw[key] = raw[key][positions]
                flows = tuple(f"{markets[i % len(markets)]}/T{i:04d}" for i in range(count))
                raw["sample_ids"] = [f"{flow}/{at}" for flow in flows]
                rows = observation_rows(engine, raw)
                cpu_rng, cuda_rng = (
                    torch.get_rng_state().clone(),
                    torch.cuda.get_rng_state(0).clone(),
                )

                extract = partial(
                    extract_call,
                    rows=rows,
                    engine=engine,
                    initial=initial,
                    flows=flows,
                    at=at,
                    dtype=dtype,
                )

                report["cases"].append(
                    dict(
                        rows=count,
                        markets=list(markets),
                        dtype=str(dtype),
                        model_id=engine.model_id,
                        configuration=model.config.identity(),
                        extract_and_seal=measure(
                            partial(extract, transfer=False),
                            warmup=args.warmup,
                            repetitions=args.repetitions,
                            rows=count,
                        ),
                        extract_seal_to_cuda=measure(
                            partial(extract, transfer=True),
                            warmup=args.warmup,
                            repetitions=args.repetitions,
                            rows=count,
                        ),
                    )
                )
                assert torch.equal(torch.get_rng_state(), cpu_rng)
                assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
                assert initial.cutoff is None and initial.calibration.pending_rows == 0
                print(json.dumps(dict(rows=count, markets=markets, dtype=str(dtype))), flush=True)
    report.update(
        process_seconds=time.perf_counter() - start,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        scope=(
            "Fixtures preparados en CPU, consumidor y contexto inicial en CUDA. La medida "
            "incluye verificación del consumidor, comprobación de estado y filas, huellas, "
            "RMS FP64, agregación canónica y sellado. La segunda medida incluye además la "
            "transferencia del vector al dispositivo. Construcción fuera de las latencias "
            "y dentro del proceso. No incluye codificadores externos, predicción, selector, "
            "feedback o publicación de una sesión financiera. Orden fijo. Sin pruebas ni "
            "perfiles propios concurrentes. No se mide energía."
        ),
        sources={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                ROOT / "src/mars_titan/memory/mechanism_observations.py",
                ROOT / "src/mars_titan/models/titans/mechanism_observation.py",
                ROOT / "src/mars_titan/models/titans/mechanism_consumer.py",
                ROOT / "tests/memory/test_mechanism_observations.py",
                ROOT / "tests/models/titans/test_mechanism_consumer.py",
                ROOT / "tests/models/titans/test_financial_adapter.py",
            )
        },
    )
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
