"""Coste técnico CPU de E3 en la sesión durable, con fuentes manuales verificadas.

Compara la sesión con corrección madura proximal sin E3, con E3 y cupo completo (que debe
conservar el estado de la referencia) y con los tres selectores a cupo uno por mercado.
Cada caso recorre tres eventos de preparación, cierre, reapertura, el evento cronometrado,
cierre y reapertura otra vez, y la siguiente transición. No ejecuta pasos de optimizador.
"""

import argparse
import gc
import hashlib
import json
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

CONTROLS = ("none", "full_quota", "standardized_error", "absolute_error", "uniform")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(cases):
    """Medianas por tamaño y control, sin contar los calentamientos declarados."""
    measured = [row for row in cases if not row["warmup"]]
    result = []
    for assets in sorted({row["assets"] for row in measured}):
        reference = None
        for control in CONTROLS:
            values = [
                row["step_seconds"]
                for row in measured
                if (row["assets"], row["control"]) == (assets, control)
            ]
            if not values:
                raise ValueError("Falta un control en la medida")
            median = statistics.median(values)
            reference = median if control == "none" else reference
            result.append(
                dict(
                    assets=assets,
                    control=control,
                    samples=len(values),
                    minimum_seconds=min(values),
                    median_seconds=median,
                    maximum_seconds=max(values),
                    relative_to_none_percent=100 * (median - reference) / reference,
                )
            )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=900)
    args = parser.parse_args()
    root, out = args.repository.resolve(), args.output.resolve()
    if (
        out.exists()
        or out.is_relative_to(root)
        or any(x not in (4, 64, 256, 1024) for x in args.sizes)
        or not 0 <= args.warmup <= 3
        or not 1 <= args.repeats <= 15
        or not 30 <= args.seconds <= 3600
    ):
        raise ValueError("La salida debe ser nueva y el presupuesto debe ser acotado")
    for part in ("src", ".", "tests/memory", "tests/models/titans"):
        sys.path.insert(0, str(root / part))
    import numpy as np
    import pyarrow as pa
    import torch
    from test_financial_session_associative import label, moment, options
    from test_financial_session_controls import flow_source, no_target_estimation

    from benchmarks.regional_decision_cuda import proof
    from mars_titan.data.storage import atomic_json
    from mars_titan.hardware.platform_identity import cpu_name
    from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
    from mars_titan.memory.financial_session import FinancialSession
    from mars_titan.memory.native_backend import load_native
    from mars_titan.memory.salience_budget import SalienceBudgetConfig
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
    from mars_titan.training.learning_hold import (
        hold_path,
        install_optimizer_guard,
        learning_blocked,
    )

    if torch.cuda.is_initialized():
        raise ValueError("Esta medida CPU no comparte una carga CUDA inicializada")
    hold = hold_path()
    if not learning_blocked() or hold.is_symlink():
        raise ValueError("La prueba técnica requiere el bloqueo de aprendizaje")
    hold_sha = digest(hold)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)
    native = load_native(args.native)
    out.mkdir(parents=True)
    started = time.perf_counter()
    deadline = time.monotonic() + args.seconds
    optimizer_attempts = []

    def forbidden(reason):
        optimizer_attempts.append(reason)
        raise RuntimeError(reason)

    guard = install_optimizer_guard(forbidden, path=hold)
    assert guard is not None
    correction = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25))
    # Cupo uno por mercado para que la selección descarte escrituras con cualquier tamaño.
    factors = dict(
        none=None,
        full_quota=SalienceBudgetConfig(writes_per_market=8192, min_history=1),
        standardized_error=SalienceBudgetConfig(writes_per_market=1, min_history=1),
        absolute_error=SalienceBudgetConfig(
            rule="absolute_error", writes_per_market=1, min_history=1
        ),
        uniform=SalienceBudgetConfig(rule="uniform", writes_per_market=1, min_history=1),
    )
    report = dict(
        schema_version=1,
        kind="salience_durable_session_technical_cost",
        status="running",
        measured_at_utc=datetime.now(UTC).isoformat(),
        protocol=dict(
            sizes=args.sizes,
            controls=list(CONTROLS),
            factors={k: v.identity() if v else None for k, v in factors.items()},
            correction=correction.memory.rule,
            warmup=args.warmup,
            repeats=args.repeats,
            device="cpu",
            dtype="torch.float64",
            block_rows=64,
            prefix_events=[125, 126, 127],
            timed_event=128,
            next_recovery_event=130,
            child_budget_seconds=args.seconds,
        ),
        hold_sha256=hold_sha,
        script_sha256=digest(Path(__file__)),
        native_sha256=digest(args.native),
        versions=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            pyarrow=pa.__version__,
        ),
        git_base_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        hardware=dict(
            cpu=cpu_name(),
            platform=platform.platform(),
            device="cpu",
            cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            gpu_used=False,
            vram_peak_bytes=None,
        ),
        cases=[],
        load_start=os.getloadavg(),
        limits=[
            "Solo fixtures manuales con modalidades compactas y un padre sin entrenar.",
            "No es rendimiento sobre el corpus completo ni una medida de precisión.",
            "No usa GPU, no mide energía y no enfría la caché del sistema de archivos.",
            "No se estiman p95/p99 fiables con tres repeticiones.",
            "El pico RSS corresponde a todo el proceso, incluida preparación y comprobaciones.",
        ],
    )
    atomic_json(out / "protocol.json", report["protocol"])

    def check():
        if time.monotonic() > deadline:
            raise TimeoutError("La prueba técnica excedió su presupuesto")
        if digest(hold) != hold_sha or optimizer_attempts:
            raise RuntimeError("Cambió la protección o hubo un intento de optimizador")

    def run_session(source, consumer, directory, factor):
        settings = options(native, source, consumer, associative=correction)
        settings.update(block_rows=64, max_log_bytes=512 * 1024**2, salience=factor)
        record, run = {}, None
        cpu_rng = torch.get_rng_state().clone()
        try:
            run = FinancialSession(directory, **settings)
            pending = []
            for event in (125, 126, 127):
                feedback = [
                    native.Feedback(p.id, 0, moment(event), label(p.asset, event)) for p in pending
                ]
                pending = run.step(source["batches"][event], feedback, cutoff=moment(event))
                pending = pending.predictions
            before = run.snapshot()
            run.close()
            run = FinancialSession(directory, **settings, resume=True)
            assert run.snapshot() == before
            feedback = [native.Feedback(p.id, 0, moment(128), label(p.asset, 128)) for p in pending]
            tick = time.perf_counter()
            result = run.step(source["batches"][128], feedback, cutoff=moment(128))
            record["step_seconds"] = time.perf_counter() - tick
            bundle = run._bundle(run.snapshot()["state"])
            memory = run._associative_state(bundle["associative"])[0].export()
            snapshot = run.snapshot()
            common = dict(
                emitted=sorted((p.asset, p.decision_at, p.value) for p in result.predictions),
                memory={k: memory[k] for k in ("writes", "cursor", "matrix_sha256")},
                native={k: snapshot[k] for k in ("applied", "cursor", "issued", "observed")},
            )
            record["common_state_sha256"] = proof(torch, common)[0]
            record["memory_writes"] = memory["writes"]
            record["applied"] = snapshot["applied"]
            record["rows"] = len(result.predictions)
            state = run.salience_state()
            record["salience_selected"] = None if state is None else state["statistics"]["selected"]
            tick = time.perf_counter()
            run.close()
            run = FinancialSession(directory, **settings, resume=True)
            assert run.snapshot() == snapshot and run.salience_state() == state
            record["recovery_seconds"] = time.perf_counter() - tick
            paths = [p for p in directory.rglob("*") if p.is_file()]
            record["persisted_bytes"] = sum(p.stat().st_size for p in paths)
            feedback = [
                native.Feedback(p.id, 0, moment(130), label(p.asset, 130))
                for p in result.predictions
            ]
            following = run.step(source["batches"][130], feedback, cutoff=moment(130))
            record["next_state_sha256"] = proof(
                torch,
                dict(
                    snapshot=run.snapshot(),
                    salience=run.salience_state(),
                    predictions=sorted(
                        (p.asset, p.decision_at, p.value) for p in following.predictions
                    ),
                ),
            )[0]
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert all(p.grad is None for p in consumer.predictor.parameters())
            consumer.verify()
            return record
        finally:
            if run is not None:
                run.close()

    def alarm(*_):
        raise TimeoutError("La prueba técnica agotó su límite de proceso")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    try:
        with contextmanager(no_target_estimation.__wrapped__)():
            for rows in args.sizes:
                check()
                flows = tuple(f"US/T{i:04d}" for i in range(rows))
                source = flow_source(out / f"fixture-{rows}", flows=flows)
                config = FinancialConfig(
                    source["spec"], variant="mac_online", hidden_size=32, seed=42
                )
                consumer = FrozenFinancialConsumer(
                    FinancialPredictor(config, dtype=torch.float64, device="cpu")
                    .eval()
                    .requires_grad_(False)
                )
                model_sha = proof(torch, consumer.predictor.state_dict())[0]
                states, next_states = {}, {}
                for index in range(args.warmup + args.repeats):
                    # El orden de los controles rota en cada repetición.
                    shift = index % len(CONTROLS)
                    for control in CONTROLS[shift:] + CONTROLS[:shift]:
                        check()
                        record = run_session(
                            source, consumer, out / f"{rows}-{control}-{index}", factors[control]
                        )
                        record.update(
                            assets=rows,
                            control=control,
                            repetition=index,
                            warmup=index < args.warmup,
                        )
                        for seen, key in (
                            (states, "common_state_sha256"),
                            (next_states, "next_state_sha256"),
                        ):
                            if control in seen:
                                assert record[key] == seen[control], "El caso no se reproduce"
                            seen[control] = record[key]
                        report["cases"].append(record)
                        atomic_json(out / "measurement.json", report)
                        print(
                            json.dumps(
                                {
                                    k: record[k]
                                    for k in (
                                        "assets",
                                        "control",
                                        "step_seconds",
                                        "memory_writes",
                                        "applied",
                                    )
                                }
                            ),
                            flush=True,
                        )
                # Con cupo completo E3 debe reproducir exactamente la referencia sin E3.
                assert states["full_quota"] == states["none"], "El cupo completo cambió el estado"
                assert proof(torch, consumer.predictor.state_dict())[0] == model_sha
                del source, consumer
                gc.collect()
        check()
        report.update(
            status="completed",
            summary=summarize(report["cases"]),
            full_quota_matches_reference=True,
            elapsed_seconds=time.perf_counter() - started,
            optimizer_attempts=optimizer_attempts,
            load_end=os.getloadavg(),
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        )
    except BaseException as error:
        report.update(
            status="failed",
            error_type=type(error).__name__,
            error=str(error),
            elapsed_seconds=time.perf_counter() - started,
        )
        raise
    finally:
        guard.remove()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        atomic_json(out / "measurement.json", report)


if __name__ == "__main__":
    main()
