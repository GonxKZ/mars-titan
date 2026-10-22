"""Coste de alternativas condicionales en CUDA, con controles fijos y sin aprendizaje."""

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
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]

from test_episodic_readout import CONTEXT  # noqa: E402
from test_market_memory import KINDS  # noqa: E402
from test_mechanism_collection import many_rows  # noqa: E402
from test_mechanism_consumer import calibration, consumer  # noqa: E402
from test_mechanism_observations import AT, DAY  # noqa: E402

from mars_titan.calibration.cohort_feedback import CohortOutcome  # noqa: E402
from mars_titan.data.input_policy import MODALITIES  # noqa: E402
from mars_titan.memory.mechanism_collection import MechanismCollector  # noqa: E402
from mars_titan.memory.mechanism_supervision import MechanismSupervision  # noqa: E402
from mars_titan.models.titans.config import canonical  # noqa: E402
from mars_titan.models.titans.financial_inputs import DecisionBatch  # noqa: E402

MODES = ("fixed36", "collect_seal", "collect_receive_metrics")


def run_case(engine, rows, mode):
    flows = tuple(r.flow_id for r in rows)
    fast, state = engine.predictor.initial_state(flows), engine.initial_state()
    if mode == "fixed36":
        raw = dict(
            inputs={name: np.stack([r.inputs[name] for r in rows]) for name in MODALITIES},
            presence=np.stack([r.presence for r in rows]),
            sample_ids=[r.sample_id for r in rows],
            prediction_at=np.full(len(rows), AT, dtype="datetime64[us]"),
            input_available_at=np.array(
                [r.input_available_at for r in rows], dtype="datetime64[us]"
            ),
        )
        batch = DecisionBatch.from_corpus(
            raw, engine.predictor.config.inputs, device="cuda:0", dtype=torch.float64
        )
        values, work = [], dict(core_observations=0, reader_refinements=0)
        for action in engine.catalog.actions():
            event = engine.begin(
                state,
                flow_ids=flows,
                cutoff=AT,
                context_id=CONTEXT,
                matures_at=(AT + DAY,) * len(rows),
                action=action,
            )
            prepared = engine.prepare(batch, fast, event)
            emitted = engine.finish(prepared.event)
            values.append(emitted.quantiles.tolist())
            for key in work:
                work[key] += emitted.work[key]
        return values, 0, work
    builder = MechanismCollector(
        engine, state, rows=rows, cutoff=AT, context_id=CONTEXT, matures_at=(AT + DAY,) * len(rows)
    )
    builder.add(rows, fast)
    packet = builder.finish()
    body = packet.payload()
    if mode == "collect_receive_metrics":
        labels = [CohortOutcome("US", AT, row.sample_id, AT + DAY, value=0.125) for row in rows]
        supervised = MechanismSupervision.start(packet).receive(labels, at=AT + DAY)
        metrics = supervised.metrics(device="cuda:0")
        if metrics[0]["mature_rows"] != len(rows):
            raise RuntimeError("La recepción perdió una etiqueta manual del fixture")
    candidates = body["groups"][0]["candidates"]
    if not all(c["status"] == "observed" for c in candidates):
        raise RuntimeError("Esta medida exige las 36 alternativas observadas")
    return [c["quantiles"] for c in candidates], len(packet.record()["payload_utf8"]), body["work"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=4)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not args.output.parent.is_dir()
        or (
            not 1 <= args.warmup <= 4
            or not 2 <= args.repetitions <= 16
            or len(set(args.rows)) != len(args.rows)
            or any(n not in (4, 64, 256) for n in args.rows)
        )
    ):
        raise ValueError("La medida necesita una salida nueva y repeticiones acotadas")
    if not torch.cuda.is_available():
        raise RuntimeError("La medida necesita cuda:0 y no pasa a CPU")
    torch.cuda.set_device("cuda:0")
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    started = time.perf_counter()
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
        dtype="float64",
        warmup=args.warmup,
        repetitions=args.repetitions,
        hidden_size=32,
        input_shapes=dict(prices=[64, 5], news=2, charts=2, fundamentals=3, macro=3),
        real_corpus=False,
        optimizer_steps=0,
        selector_fitted=False,
        scientific_study=False,
        case_order="rotating_three_modes_each_repetition",
        cases=[],
    )
    for size in args.rows:
        engine = consumer("cuda:0", kind=KINDS[0], reader=True, calibrator=calibration())
        rows = many_rows(engine, size)
        samples = {mode: [] for mode in MODES}
        peaks = {mode: dict(allocated=0, reserved=0) for mode in MODES}
        digests = {mode: set() for mode in MODES}
        sizes, works, differences = {}, {}, dict.fromkeys(MODES, 0.0)
        reference = None
        cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(0)
        for index in range(args.warmup + args.repetitions):
            order = MODES[index % 3 :] + MODES[: index % 3]
            for mode in order:
                torch.cuda.synchronize(0)
                torch.cuda.reset_peak_memory_stats(0)
                before = time.perf_counter_ns()
                quantiles, retained, work = run_case(engine, rows, mode)
                torch.cuda.synchronize(0)
                elapsed = (time.perf_counter_ns() - before) / 1e6
                values = np.asarray(quantiles, dtype=np.float64)
                if reference is None:
                    reference = values.copy()
                np.testing.assert_allclose(values, reference, rtol=1e-12, atol=1e-12)
                differences[mode] = max(
                    differences[mode], float(np.max(np.abs(values - reference)))
                )
                if index >= args.warmup:
                    samples[mode].append(elapsed)
                    sizes[mode], works[mode] = retained, work
                    digests[mode].add(hashlib.sha256(canonical(quantiles).encode()).hexdigest())
                    peaks[mode]["allocated"] = max(
                        peaks[mode]["allocated"], torch.cuda.max_memory_allocated(0)
                    )
                    peaks[mode]["reserved"] = max(
                        peaks[mode]["reserved"], torch.cuda.max_memory_reserved(0)
                    )
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
            cuda_rng, torch.cuda.get_rng_state(0)
        ):
            raise RuntimeError("La colección modificó el RNG externo")
        for mode in MODES:
            times = samples[mode]
            if len(digests[mode]) != 1:
                raise RuntimeError("Una repetición cambió las alternativas")
            report["cases"].append(
                dict(
                    rows=size,
                    mode=mode,
                    samples_ms=times,
                    **{f"p{p}_ms": float(np.percentile(times, p)) for p in (50, 95, 99)},
                    mean_ms=float(np.mean(times)),
                    std_ms=float(np.std(times, ddof=1)),
                    asset_cohorts_per_second=1000 * size / float(np.mean(times)),
                    packet_bytes=sizes[mode],
                    work=works[mode],
                    allocator_peaks_bytes=peaks[mode],
                    quantiles_sha256=next(iter(digests[mode])),
                    max_abs_difference=differences[mode],
                    consumer_identity=engine.identity(),
                )
            )
        print(json.dumps(dict(rows=size, completed=True)), flush=True)
    report["process_seconds"] = time.perf_counter() - started
    report["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    report["scope"] = (
        "Fixtures US sin banco y desde contexto inicial, dimensiones pequeñas y dos contextos. "
        "fixed36 es una referencia de 36 ejecuciones fijas, no un ejecutor óptimo. No extrae "
        "la observación ni sella un paquete. collect_seal incluye extracción, bloques, "
        "predicciones y calibración condicional, validación y sellado. La tercera modalidad "
        "añade recepción de etiquetas manuales y métricasGPU, sin ajuste. Los tiempos son "
        "inclusivos y no se suman. No son latencia de despliegue ni incluyen publicación "
        "en una generación nativa, disco, corpus real, codificadores externos o energía. "
        "Percentiles empíricos de cuatro medidas, sin intervalo estadístico."
    )
    paths = [
        Path(__file__),
        *(
            ROOT / name
            for name in (
                "src/mars_titan/memory/mechanism_collection.py",
                "src/mars_titan/memory/mechanism_collection_record.py",
                "src/mars_titan/memory/mechanism_supervision.py",
                "src/mars_titan/models/titans/mechanism_selection.py",
                "tests/memory/test_mechanism_collection.py",
                "tests/models/titans/test_mechanism_consumer.py",
            )
        ),
    ]
    report["sources"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
    }
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
