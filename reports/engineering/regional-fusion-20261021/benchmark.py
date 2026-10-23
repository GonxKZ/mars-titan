"""Coste de fusión y consumidor CUDA con datos técnicos residentes, sin entrenar."""

import argparse
import hashlib
import json
import platform
import resource
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests/models/titans"))

from test_financial_adapter import raw_batch, specification  # noqa: E402

from mars_titan.data.price_windows import gate_price_window  # noqa: E402
from mars_titan.models.baselines.multimodal import fuse_modalities  # noqa: E402
from mars_titan.models.titans.financial import DecisionBatch, _tensor_digest  # noqa: E402
from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer  # noqa: E402
from mars_titan.training.titans_walk_forward import _predictor  # noqa: E402


def source_batch(rows, spec, dtype):
    raw = raw_batch()
    indices = np.arange(rows) % 2
    raw["inputs"] = {key: value[indices] for key, value in raw["inputs"].items()}
    raw["presence"] = raw["presence"][indices]
    raw["sample_ids"] = [f"US/T{index:04d}/1609459200000000" for index in range(rows)]
    raw["market"] = ["US"] * rows
    for key in ("prediction_at", "input_available_at"):
        raw[key] = raw[key][indices]
    return DecisionBatch.from_corpus(raw, spec, device="cuda:0", dtype=dtype)


def measure(call, *, rows, warmup, repetitions):
    samples, digests = [], []
    torch.cuda.reset_peak_memory_stats(0)
    with torch.no_grad():
        for index in range(warmup + repetitions):
            torch.cuda.synchronize(0)
            start = time.perf_counter_ns()
            output = call()
            torch.cuda.synchronize(0)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if index >= warmup:
                samples.append(elapsed)
                value = output.point_predictions if hasattr(output, "point_predictions") else output
                digests.append(_tensor_digest(value))
            del output
    if len(set(digests)) != 1:
        raise RuntimeError("Las repeticiones no conservan la misma salida")
    return dict(
        samples_ms=samples,
        **{f"p{p}_ms": float(np.percentile(samples, p)) for p in (50, 95, 99)},
        mean_ms=float(np.mean(samples)),
        std_ms=float(np.std(samples, ddof=1)),
        rows_per_second=rows * 1000 / float(np.mean(samples)),
        output_sha256=digests[0],
        allocated_vram_peak_bytes=torch.cuda.max_memory_allocated(0),
        reserved_vram_peak_bytes=torch.cuda.max_memory_reserved(0),
    )


def case(recipe, changes, *, rows, dtype, warmup, repetitions):
    spec = specification()
    model, pairing = _predictor(recipe, spec, "mac_online", 42, "cuda:0", changes=changes or None)
    assert pairing is None
    model = model.to(dtype=dtype).eval().requires_grad_(False)
    engine = FrozenFinancialConsumer(model)
    batch = source_batch(rows, spec, dtype)
    initial = model.initial_state(batch.flow_ids)
    cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    with torch.no_grad():
        price, checks = model.price_encoder.encode_last(gate_price_window(batch.inputs["prices"]))
        assert all(torch.isfinite(value).all() for value, _ in checks)
    fusion = measure(
        lambda: fuse_modalities(
            price,
            batch.inputs,
            model.encoders,
            model.fusion,
            batch.presence,
            modulation=model.regional_fusion,
        ),
        rows=rows,
        warmup=warmup,
        repetitions=repetitions,
    )
    frozen = measure(
        lambda: engine.prepare(batch, initial, context_id="c" * 64),
        rows=rows,
        warmup=warmup,
        repetitions=repetitions,
    )
    engine.verify()
    assert initial.observed_steps.tolist() == [0] * rows
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(0), cuda_rng)
    common = {
        name: _tensor_digest(value)
        for name, value in model.named_parameters()
        if not name.startswith("regional_fusion.")
    }
    return dict(
        rows=rows,
        dtype=str(dtype),
        initialization_dtype="torch.float32_then_cast",
        model_id=engine.model_id,
        configuration=model.config.identity(),
        parameters=sum(value.numel() for value in model.parameters()),
        parameter_bytes=sum(value.numel() * value.element_size() for value in model.parameters()),
        shared_parameter_sha256=hashlib.sha256(
            json.dumps(common, sort_keys=True).encode()
        ).hexdigest(),
        persistent_tensor_bytes_per_flow=model._tensor_bytes_per_flow(),
        regional_work=model.regional_fusion.operation_counts(rows)
        if model.regional_fusion is not None
        else {},
        fusion=fusion,
        frozen_consumer=frozen,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=12)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.output.parent.is_dir()
        or not 1 <= args.warmup <= 16
        or not 2 <= args.repetitions <= 128
        or len(set(args.rows)) != len(args.rows)
        or any(n not in (64, 256) for n in args.rows)
    ):
        raise ValueError("La medida necesita salida nueva, tamaños y repeticiones válidos")
    if not torch.cuda.is_available():
        raise RuntimeError("Esta medida requiere cuda:0 y no pasa a CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    arms_path = ROOT / "configs/experiments/regional-fusion-v1.json"
    arms = json.loads(arms_path.read_text())
    recipe_path = ROOT / arms["base_recipe"]
    recipe = json.loads(recipe_path.read_text())
    recipe["predictor"]["max_batch"] = 256
    started = time.perf_counter()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        device="cuda:0",
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu_driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        numpy=np.__version__,
        cpu_threads=torch.get_num_threads(),
        warmup=args.warmup,
        repetitions=args.repetitions,
        seed=42,
        benchmark_overrides=dict(max_batch=256),
        real_data_used=False,
        optimizer_steps=0,
        scientific_study=False,
        cases=[],
    )
    for dtype in (torch.float32, torch.float64):
        for rows in args.rows:
            reference = None
            for name, arm in arms["arms"].items():
                measured = case(
                    recipe,
                    arm["predictor_changes"],
                    rows=rows,
                    dtype=dtype,
                    warmup=args.warmup,
                    repetitions=args.repetitions,
                )
                if reference is None:
                    reference = measured["shared_parameter_sha256"]
                assert measured["shared_parameter_sha256"] == reference
                report["cases"].append(dict(arm=name, **measured))
                print(
                    json.dumps(dict(arm=name, rows=rows, dtype=str(dtype), completed=True)),
                    flush=True,
                )
    report.update(
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        scope=(
            "Fixtures residentes en CUDA. Fusión incluye las proyecciones de las otras fuentes "
            "y excluye el codificador de precios. Consumidor incluye codificador, fusión, MAC "
            "online y cabeza, con controles congelados. Cada repetición parte del mismo estado "
            "inicial inmutable y avanza una observación. Construcción, transferencias iniciales "
            "y preparación del estado quedan fuera de las latencias y dentro del proceso. "
            "No incluye journal, codificadores externos, entrenamiento ni corpus real. "
            "Orden fijo y percentiles empíricos, sin energía medida."
        ),
        sources={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                arms_path,
                recipe_path,
                ROOT / "src/mars_titan/models/titans/regional_fusion.py",
                ROOT / "src/mars_titan/models/titans/financial.py",
                ROOT / "src/mars_titan/models/titans/frozen_financial.py",
                ROOT / "src/mars_titan/models/baselines/multimodal.py",
                ROOT / "tests/models/titans/test_financial_adapter.py",
            )
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(report["cases"]), seconds=report["process_seconds"])))


if __name__ == "__main__":
    main()
