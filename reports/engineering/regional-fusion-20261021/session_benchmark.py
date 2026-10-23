"""Coste de sesiones selladas con y sin grafo, incluidos transferencias y archivo."""

import argparse
import hashlib
import importlib.util
import json
import os
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

from test_financial_mechanisms import engines  # noqa: E402
from test_financial_session_controls import flow_source  # noqa: E402
from test_regional_fusion_session import regional_data  # noqa: E402

from mars_titan.memory.native_backend import load_native  # noqa: E402
from mars_titan.models.titans.financial import _tensor_digest  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=6)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.output.parent.is_dir()
        or not args.storage_root.is_dir()
        or not 1 <= args.warmup <= 16
        or not 2 <= args.repetitions <= 128
        or len(set(args.rows)) != len(args.rows)
        or any(n not in (4, 64, 256) for n in args.rows)
    ):
        raise ValueError("La medida necesita salida nueva, almacenamiento y tamaños válidos")
    if not torch.cuda.is_available():
        raise RuntimeError("La sesión requiere cuda:0 y no pasa a CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    native = load_native(os.environ["MARS_TITAN_EPISODIC_NATIVE"])
    native.require_safe_path(str(args.storage_root))
    reference_path = ROOT / "reports/engineering/mechanism-session-20261021/benchmark.py"
    spec = importlib.util.spec_from_file_location("regional_session_trajectory", reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    started = time.perf_counter()
    report = dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        device="cuda:0",
        gpu=torch.cuda.get_device_name(0),
        gpu_total_bytes=torch.cuda.get_device_properties(0).total_memory,
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        native_sha256=native.binary_sha256,
        dtype="float64",
        hidden_size=32,
        block_rows=64,
        bank_capacity=64,
        modalities=dict(prices=[64, 5], news=2, charts=1, fundamentals=3, macro=3),
        warmup=args.warmup,
        repetitions=args.repetitions,
        real_data_used=False,
        optimizer_steps=0,
        scientific_study=False,
        cases=[],
    )
    with tempfile.TemporaryDirectory(prefix="regional-session-", dir=args.storage_root) as folder:
        directory = Path(folder)
        for rows in args.rows:
            source = flow_source(
                directory / f"source-{rows}", flows=tuple(f"US/T{i:04d}" for i in range(rows))
            )
            common_reference = None
            for arm in ("original_context_online", "regional_ring_context_online"):
                data = (
                    engines(source, "cuda:0", online=True)
                    if arm == "original_context_online"
                    else regional_data(source, "cuda:0")
                )
                engine = data[3]
                common = {
                    name: _tensor_digest(value)
                    for name, value in engine.predictor.named_parameters()
                    if not name.startswith("regional_fusion.")
                }
                common_id = hashlib.sha256(json.dumps(common, sort_keys=True).encode()).hexdigest()
                if common_reference is None:
                    common_reference = common_id
                assert common_id == common_reference
                samples, sizes, digests = [], [], []
                torch.cuda.reset_peak_memory_stats(0)
                for index in range(args.warmup + args.repetitions):
                    times, size, digest = reference.trajectory(
                        native, source, engine, directory / f"run-{rows}-{arm}-{index}"
                    )
                    if index >= args.warmup:
                        samples.append(times)
                        sizes.append(size)
                        digests.append(digest)
                assert len(set(digests)) == 1
                totals = [item["total"] for item in samples]
                report["cases"].append(
                    dict(
                        arm=arm,
                        rows=rows,
                        model_id=engine.model_id,
                        configuration=engine.predictor.config.identity(),
                        shared_parameter_sha256=common_id,
                        parameters=sum(p.numel() for p in engine.predictor.parameters()),
                        samples_ms=samples,
                        **{f"p{p}_ms": float(np.percentile(totals, p)) for p in (50, 95, 99)},
                        mean_ms=float(np.mean(totals)),
                        std_ms=float(np.std(totals, ddof=1)),
                        prediction_rows_per_second=3 * rows * 1000 / float(np.mean(totals)),
                        retained_bytes=sizes,
                        prediction_sha256=digests[0],
                        allocated_vram_peak_bytes=torch.cuda.max_memory_allocated(0),
                        reserved_vram_peak_bytes=torch.cuda.max_memory_reserved(0),
                    )
                )
                print(json.dumps(dict(arm=arm, rows=rows, completed=True)), flush=True)
    report.update(
        process_seconds=time.perf_counter() - started,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        scope=(
            "Fixtures con precios replicados y etiquetas manuales. Una apertura, calentamiento, "
            "tres emisiones, dos recepciones de resultados, cierre censurado, escritura sellada, "
            "poda, lectura de una traza antigua y recuperación. Las dos configuraciones comparten "
            "contexto PT4, calibración PT2, memoria, parámetros del tronco y archivo. Solo el "
            "segundo añade fusión regional en anillo. Incluye las transferencias durante cada "
            "sesión. Construcción fuera de las latencias y dentro del proceso. No incluye "
            "codificadores externos, corpus real o aprendizaje. Orden fijo y percentiles "
            "empíricos, sin energía medida."
        ),
        sources={
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                reference_path,
                ROOT / "src/mars_titan/models/titans/regional_fusion.py",
                ROOT / "src/mars_titan/models/titans/financial.py",
                ROOT / "src/mars_titan/models/titans/mechanism_consumer.py",
                ROOT / "src/mars_titan/memory/financial_mechanisms.py",
                ROOT / "src/mars_titan/memory/financial_session.py",
                ROOT / "tests/memory/test_regional_fusion_session.py",
                ROOT / "tests/memory/test_financial_session_controls.py",
            )
        },
    )
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps(dict(cases=len(report["cases"]), seconds=report["process_seconds"])))


if __name__ == "__main__":
    main()
