"""Coste técnico CPU de sesiones B4 durables con fuentes manuales verificadas."""

import argparse
import cProfile
import gc
import hashlib
import json
import math
import os
import platform
import pstats
import resource
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def io_counters():
    """Contadores del proceso en Linux, sin atribuirlos a E/S física del dispositivo."""
    path = Path("/proc/self/io")
    if not path.is_file():
        return None
    with path.open() as stream:
        return {
            key: int(value)
            for key, value in (line.split(":", 1) for line in stream.read(4096).splitlines())
        }


def cost_summary(cases):
    """Resumir muestras emparejadas, excluyendo los calentamientos declarados."""
    measured = [row for row in cases if not row["warmup"]]
    if not measured or any(
        type(row["step_seconds"]) not in (int, float)
        or not math.isfinite(row["step_seconds"])
        or row["step_seconds"] <= 0
        for row in measured
    ):
        raise ValueError("Las duraciones medidas deben ser finitas y positivas")
    summaries = []
    for assets, profile in sorted({(row["assets"], row["profile"]) for row in measured}):
        paired, stats = {}, {}
        for active, name in ((False, "original"), (True, "diagnostic")):
            rows = [
                row
                for row in measured
                if (row["assets"], row["profile"], row["diagnostics"]) == (assets, profile, active)
            ]
            indexed = {row["repetition"]: row["common_state_sha256"] for row in rows}
            if not rows or len(indexed) != len(rows):
                raise ValueError("Faltan controles o hay repeticiones duplicadas")
            paired[active] = indexed
            values = [row["step_seconds"] for row in rows]
            median = statistics.median(values)
            stats[name] = dict(
                samples=len(values),
                minimum_seconds=min(values),
                median_seconds=median,
                maximum_seconds=max(values),
                rows_per_second_at_median=assets / median,
            )
        if paired[False] != paired[True]:
            raise ValueError("Las comparaciones no conservan repeticiones y estados originales")
        delta = stats["diagnostic"]["median_seconds"] - stats["original"]["median_seconds"]
        summaries.append(
            dict(
                assets=assets,
                profile=profile,
                **stats,
                delta_median_seconds=delta,
                relative_extra_percent=100 * delta / stats["original"]["median_seconds"],
            )
        )
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64])
    parser.add_argument(
        "--profiles", nargs="+", choices=["areas", "bridge", "multiscale"], default=["bridge"]
    )
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--seconds", type=int, default=180)
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
    from mars_titan.memory.area_memory import AreaLayout
    from mars_titan.memory.area_query_bridge import AreaQueryBridgeConfig
    from mars_titan.memory.associative_memory import AssociativeMemoryConfig, MatureCorrection
    from mars_titan.memory.financial_session import FinancialSession
    from mars_titan.memory.lesion_protocol import LesionProtocol
    from mars_titan.memory.multiscale_mixture import MultiscaleMixtureConfig
    from mars_titan.memory.native_backend import load_native
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
    base = MatureCorrection(AssociativeMemoryConfig("proximal", rate=0.25), areas=AreaLayout())
    bridge = replace(base, query_bridge=AreaQueryBridgeConfig())
    profiles = dict(
        areas=base,
        bridge=bridge,
        multiscale=replace(
            bridge,
            memory=replace(bridge.memory, forgetting=0.0),
            multiscale=MultiscaleMixtureConfig(),
        ),
    )
    report = dict(
        schema_version=1,
        kind="lesion_durable_session_technical_cost",
        status="running",
        measured_at_utc=datetime.now(UTC).isoformat(),
        protocol=dict(
            sizes=args.sizes,
            profiles=args.profiles,
            warmup=args.warmup,
            repeats=args.repeats,
            device="cpu",
            dtype="torch.float64",
            block_rows=64,
            prefix_events=[125, 126, 127],
            timed_event=128,
            next_recovery_event=130,
            mask_conditions=20,
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
        source_fixtures=[],
        load_start=os.getloadavg(),
        limits=[
            "Solo fixtures manuales con modalidades compactas y presencia de precios y gráficos.",
            "No es rendimiento sobre el corpus completo ni de un modelo entrenado.",
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

    def inputs_proof(source):
        h, total = hashlib.sha256(), 0
        for at in sorted(source["batches"]):
            for batch in source["batches"][at]:
                batch.verify()
                h.update(batch.input_digest.encode())
                for name in sorted(batch.inputs):
                    h.update(name.encode())
                    h.update(batch.inputs[name].tobytes())
                    total += batch.inputs[name].nbytes
        return h.hexdigest(), total

    def run_session(
        source, consumer, correction, directory, active, instrument=False, recover_after=True
    ):
        settings = options(native, source, consumer, associative=correction)
        settings.update(block_rows=64, max_log_bytes=512 * 1024**2)
        if active:
            settings["lesions"] = LesionProtocol()
        record = {}
        run = None
        cpu_rng = torch.get_rng_state().clone()
        try:
            tick = time.perf_counter()
            run = FinancialSession(directory, **settings)
            record["create_seconds"] = time.perf_counter() - tick
            pending = []
            tick = time.perf_counter()
            for event in (125, 126, 127):
                feedback = [
                    native.Feedback(p.id, 0, moment(event), label(p.asset, event)) for p in pending
                ]
                result = run.step(source["batches"][event], feedback, cutoff=moment(event))
                pending = result.predictions
            record["prefix_seconds"] = time.perf_counter() - tick
            before = run.snapshot()
            tick = time.perf_counter()
            run.close()
            run = None
            record["prefix_close_seconds"] = time.perf_counter() - tick
            tick = time.perf_counter()
            run = FinancialSession(directory, **settings, resume=True)
            record["reopen_seconds"] = time.perf_counter() - tick
            assert run.snapshot() == before
            feedback = [native.Feedback(p.id, 0, moment(128), label(p.asset, 128)) for p in pending]
            profile = cProfile.Profile() if instrument else None
            if profile:
                profile.enable()
            counters_before = io_counters()
            tick = time.perf_counter()
            result = run.step(source["batches"][128], feedback, cutoff=moment(128))
            record["step_seconds"] = time.perf_counter() - tick
            counters_after = io_counters()
            record["proc_self_io_delta"] = (
                {key: counters_after[key] - value for key, value in counters_before.items()}
                if counters_before is not None and counters_after is not None
                else None
            )
            if profile:
                profile.disable()
                profile.dump_stats(str(directory.parent / (directory.name + ".prof")))
                with (directory.parent / (directory.name + "-profile.txt")).open("w") as stream:
                    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats(
                        "cumtime"
                    ).print_stats(40)
            tick = time.perf_counter()
            bundle = run._bundle(run.snapshot()["state"])
            memory = run._associative_state(bundle["associative"])[0].export()
            emitted = [(p.id, p.asset, p.decision_at, p.value) for p in result.predictions]
            # Los ID nativos incorporan el contrato. La paridad entre variantes
            # usa identidad de activo/decisión y el contenido de los pesos rápidos.
            flows = sorted(p.asset for p in result.predictions)
            fast_hashes = [
                proof(torch, run._fast_store.gather(run.fast_state(), flows[i : i + 64]))[0]
                for i in range(0, len(flows), 64)
            ]
            snapshot = run.snapshot()
            common = dict(
                emitted=sorted((p.asset, p.decision_at, p.value) for p in result.predictions),
                memory=memory,
                fast=fast_hashes,
                native={
                    k: snapshot[k]
                    for k in (
                        "applied",
                        "cursor",
                        "excluded",
                        "finalized",
                        "issued",
                        "last_at",
                        "last_event",
                        "observed",
                        "phase_closed",
                    )
                },
            )
            record["common_state_sha256"] = proof(torch, common)[0]
            record["repeated_state_sha256"] = proof(
                torch,
                dict(
                    snapshot=run.snapshot(),
                    common=common,
                    lesions=run.lesion_metrics() if active else None,
                ),
            )[0]
            record["applied"] = len(result.applied)
            record["rows"] = len(emitted)
            snapshot = run.snapshot()
            record["proof_seconds"] = time.perf_counter() - tick
            record["close_seconds"], record["recovery_seconds"] = None, None
            if recover_after:
                tick = time.perf_counter()
                run.close()
                run = None
                record["close_seconds"] = time.perf_counter() - tick
                tick = time.perf_counter()
                run = FinancialSession(directory, **settings, resume=True)
                assert run.snapshot() == snapshot
                record["recovery_seconds"] = time.perf_counter() - tick
            record["recovery_exact"] = True
            paths = subprocess.check_output(
                ["rg", "--files", "--hidden", "--no-ignore", "-0", str(directory)]
            ).split(b"\0")
            paths = [Path(os.fsdecode(p)) for p in paths if p]
            record["persisted_bytes"] = sum(p.stat().st_size for p in paths)
            record["persisted_files"] = len(paths)
            feedback = [
                native.Feedback(p.id, 0, moment(130), label(p.asset, 130))
                for p in result.predictions
            ]
            next_result = run.step(source["batches"][130], feedback, cutoff=moment(130))
            record["next_state_sha256"] = proof(
                torch,
                dict(
                    snapshot=run.snapshot(),
                    fast=run.fast_state(),
                    predictions=[
                        (p.id, p.asset, p.decision_at, p.value) for p in next_result.predictions
                    ],
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
                tick = time.perf_counter()
                flows = tuple(f"US/T{i:04d}" for i in range(rows))
                source = flow_source(out / f"fixture-{rows}", flows=flows)
                source_time = time.perf_counter() - tick
                config = FinancialConfig(
                    source["spec"], variant="mac_online", hidden_size=32, seed=42
                )
                tick = time.perf_counter()
                consumer = FrozenFinancialConsumer(
                    FinancialPredictor(config, dtype=torch.float64, device="cpu")
                    .eval()
                    .requires_grad_(False)
                )
                model_time = time.perf_counter() - tick
                model_sha = proof(torch, consumer.predictor.state_dict())[0]
                inputs_sha, input_bytes = inputs_proof(source)
                report["source_fixtures"].append(
                    dict(
                        rows=rows,
                        setup_seconds=source_time,
                        model_seconds=model_time,
                        model_sha256=model_sha,
                        input_specification=source["spec"].identity(),
                        input_sha256=inputs_sha,
                        input_logical_bytes=input_bytes,
                        consumer_identity=consumer.identity(),
                        label_origin=source["label_origin"],
                    )
                )
                for name in args.profiles:
                    references = {}
                    commons = {}
                    next_states = {}
                    for index in range(args.warmup + args.repeats):
                        for active in (False, True) if index % 2 == 0 else (True, False):
                            check()
                            directory = out / f"{rows}-{name}-{index}-{active}"
                            record = run_session(
                                source,
                                consumer,
                                profiles[name],
                                directory,
                                active,
                                recover_after=bool(index % 2),
                            )
                            record.update(
                                assets=rows,
                                profile=name,
                                diagnostics=active,
                                repetition=index,
                                warmup=index < args.warmup,
                            )
                            if active in references:
                                assert record["repeated_state_sha256"] == references[active]
                            references[active] = record["repeated_state_sha256"]
                            if active in next_states:
                                assert record["next_state_sha256"] == next_states[active], (
                                    "La recuperación cambió la siguiente transición"
                                )
                            next_states[active] = record["next_state_sha256"]
                            commons[active] = record["common_state_sha256"]
                            if len(commons) == 2:
                                assert commons[False] == commons[True], (
                                    "B4 cambió el estado original"
                                )
                            report["cases"].append(record)
                            atomic_json(out / "measurement.json", report)
                            print(
                                json.dumps(
                                    {
                                        k: record[k]
                                        for k in (
                                            "assets",
                                            "profile",
                                            "diagnostics",
                                            "step_seconds",
                                            "persisted_bytes",
                                            "recovery_exact",
                                        )
                                    }
                                ),
                                flush=True,
                            )
                    if args.profile:
                        record = run_session(
                            source,
                            consumer,
                            profiles[name],
                            out / f"{rows}-{name}-instrumented",
                            True,
                            instrument=True,
                        )
                        assert record["common_state_sha256"] == commons[True]
                assert inputs_proof(source) == (inputs_sha, input_bytes)
                assert proof(torch, consumer.predictor.state_dict())[0] == model_sha
                del source, consumer
                gc.collect()
        check()
        report.update(
            status="completed",
            summary=cost_summary(report["cases"]),
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
