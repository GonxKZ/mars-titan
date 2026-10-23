"""Coste de selección y publicación frente al mismo K1 fijo, sin aprendizaje."""

import argparse
import hashlib
import json
import os
import platform
import resource
import runpy
import subprocess
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
from test_financial_session_controls import flow_source  # noqa: E402
from test_mechanism_policy import policy  # noqa: E402

from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer  # noqa: E402

TRAJECTORY = ROOT / "reports/engineering/mechanism-session-20261021/benchmark.py"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--rows", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=4)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
        or not args.storage_root.is_dir()
        or not 1 <= args.warmup <= 4
        or not 2 <= args.repetitions <= 16
        or len(set(args.rows)) != len(args.rows)
        or any(n not in (4, 64, 256) for n in args.rows)
    ):
        raise ValueError("Se necesita salida nueva, almacenamiento y repeticiones acotadas")
    if not torch.cuda.is_available():
        raise RuntimeError("La medida necesita cuda:0, sin alternativa CPU implícita")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    trajectory = runpy.run_path(str(TRAJECTORY))["trajectory"]
    started = time.perf_counter()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        torch=str(torch.__version__),
        numpy=np.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        driver=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
        ).strip(),
        device="cuda:0",
        dtype="float64",
        cpu_threads=torch.get_num_threads(),
        warmup=args.warmup,
        repetitions=args.repetitions,
        hidden_size=32,
        block_rows=64,
        bank_capacity=64,
        input_shapes=dict(prices=[64, 5], news=2, charts=1, fundamentals=3, macro=3),
        native_sha256=native.binary_sha256,
        real_data_used=False,
        optimizer_steps=0,
        selector_fitted=False,
        scientific_study=False,
        cases=[],
    )
    with tempfile.TemporaryDirectory(prefix="mars-selection-cost-", dir=args.storage_root) as tmp:
        folder = Path(tmp)
        for rows in args.rows:
            source = flow_source(
                folder / f"source-{rows}", flows=tuple(f"US/T{i:04d}" for i in range(rows))
            )
            parent = engines(source, "cuda:0", online=True)[3]
            selector = policy(device="cuda:0")
            selector.point_coefficients.zero_()
            selector.interval_coefficients.zero_()
            selector.support_max.fill_(100)
            calibrator = parent.initial_state().calibration
            controls = dict(
                fixed=FrozenMechanismConsumer(parent.predictor, calibration=calibrator),
                selected=FrozenMechanismConsumer(
                    parent.predictor, calibration=calibrator, policy=selector
                ),
            )
            samples, sizes, digests = ({k: [] for k in controls} for _ in range(3))
            peaks = {k: dict(allocated=0, reserved=0) for k in controls}
            for repetition in range(args.warmup + args.repetitions):
                order = tuple(controls) if repetition % 2 == 0 else tuple(reversed(controls))
                for kind in order:
                    torch.cuda.reset_peak_memory_stats(0)
                    values, size, digest = trajectory(
                        native, source, controls[kind], folder / f"{rows}-{kind}-{repetition}"
                    )
                    if repetition >= args.warmup:
                        samples[kind].append(values)
                        sizes[kind].append(size)
                        digests[kind].append(digest)
                        peaks[kind]["allocated"] = max(
                            peaks[kind]["allocated"], torch.cuda.max_memory_allocated(0)
                        )
                        peaks[kind]["reserved"] = max(
                            peaks[kind]["reserved"], torch.cuda.max_memory_reserved(0)
                        )
            if len(set(digests["fixed"] + digests["selected"])) != 1:
                raise RuntimeError("La selección K1 cambia las predicciones del control")
            for kind, engine in controls.items():
                totals = [s["total"] for s in samples[kind]]
                report["cases"].append(
                    dict(
                        rows=rows,
                        kind=kind,
                        consumer_identity=engine.identity(),
                        prefix_policy_id=source["prefixes"].policy_id,
                        samples_ms=samples[kind],
                        retained_bytes=sizes[kind],
                        prediction_sha256=digests[kind][0],
                        **{f"p{p}_ms": float(np.percentile(totals, p)) for p in (50, 95, 99)},
                        mean_ms=float(np.mean(totals)),
                        std_ms=float(np.std(totals, ddof=1)),
                        prediction_rows_per_second=3 * rows * 1000 / float(np.mean(totals)),
                        allocator_peaks_bytes=peaks[kind],
                    )
                )
            print(json.dumps(dict(rows=rows, completed=True)), flush=True)
    report["process_seconds"] = time.perf_counter() - started
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["scope"] = (
        "Fixtures US con precios replicados, máscaras y etiquetas manuales. Reutiliza "
        "trajectory: apertura, calentamiento, tres emisiones, feedback, cierre, escritura, "
        "verificación, poda y recuperación. Incluye lectura de trazas y contabilidad de disco. "
        "K1 sin lector ni contexto activado, con módulos PT4/PT2 presentes. Selector dado con "
        "ganancias cero, no ajustado. Orden de controles alternado por repetición. No incluye "
        "encoders externos, corpus real, energía, tiempo de construcción en latencias ni "
        "un intervalo estadístico. RSS es del proceso completo y los picos CUDA incluyen "
        "ambos consumidores residentes, no la VRAM del dispositivo entero."
    )
    paths = [Path(__file__), TRAJECTORY]
    paths.extend(
        ROOT / p
        for p in (
            "src/mars_titan/models/titans/mechanism_selection.py",
            "src/mars_titan/models/titans/mechanism_consumer.py",
            "src/mars_titan/models/titans/mechanism_policy.py",
            "src/mars_titan/memory/financial_mechanisms.py",
            "src/mars_titan/memory/mechanism_observations.py",
            "tests/memory/test_financial_mechanisms.py",
            "tests/memory/test_financial_session_controls.py",
            "tests/models/titans/test_mechanism_policy.py",
        )
    )
    report["sources"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
