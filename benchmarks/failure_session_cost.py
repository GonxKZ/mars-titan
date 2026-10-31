"""Coste técnico CPU del libro S2 en la sesión completa, sin aprendizaje.

Usa fixtures manuales, lector K1, banco M1 y padres sin entrenar. Compara por
separado el predictor escalar y el consumidor de mecanismos, con y sin S2.
El intervalo medido incluye la emisión y el feedback de la cuarta decisión.
La preparación de datos, las comprobaciones y la recuperación se registran aparte.
"""

import argparse
import gc
import json
import os
import platform
import resource
import shutil
import signal
import statistics
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

FAMILIES = ("scalar", "mechanism")


def study_protocol(sizes, warmup, repeats):
    """Fijar tamaños y repeticiones acotados antes de crear cualquier ejecución."""
    if (
        type(sizes) is not list
        or not sizes
        or any(type(n) is not int or n not in (4, 64, 256) for n in sizes)
        or len(sizes) != len(set(sizes))
        or type(warmup) is not int
        or not 0 <= warmup <= 3
        or type(repeats) is not int
        or not 1 <= repeats <= 10
    ):
        raise ValueError("La medida necesita tamaños únicos admitidos y repeticiones acotadas")
    return dict(
        sizes=sizes,
        families=list(FAMILIES),
        enabled=[False, True],
        warmup=warmup,
        repeats=repeats,
        device="cpu",
        dtype="torch.float64",
        block_rows=64,
        memory_reads=1,
        admission="m1",
        episodic_capacity=64,
        failure_capacity="assets",
        failure_cache_keys=True,
        prefix_events=[125, 126, 127],
        measured_event=128,
        recovery_following_event=130,
        seed=42,
    )


def summarize(cases):
    """Comparar cada familia con su propia referencia y exigir paridad completa."""
    result = []
    for assets in sorted({row["assets"] for row in cases}):
        for family in FAMILIES:
            group = [r for r in cases if (r["assets"], r["family"]) == (assets, family)]
            paired = [
                [r["repetition"] for r in group if r["enabled"] is enabled and not r["warmup"]]
                for enabled in (False, True)
            ]
            if set(paired[0]) != set(paired[1]) or any(
                len(values) != len(set(values)) for values in paired
            ):
                raise ValueError("Las repeticiones medidas no están emparejadas")
            for field in ("common_state_sha256", "next_state_sha256"):
                if len({r[field] for r in group}) != 1:
                    raise ValueError("Cambió la emisión o el estado recuperado entre controles")
            reference = None
            for enabled in (False, True):
                rows = [r for r in group if r["enabled"] is enabled and not r["warmup"]]
                if not rows:
                    raise ValueError("Falta la referencia o una variante de la medida")
                values = [r["step_seconds"] for r in rows]
                median = statistics.median(values)
                reference = median if not enabled else reference
                result.append(
                    dict(
                        assets=assets,
                        family=family,
                        enabled=enabled,
                        samples=len(rows),
                        minimum_seconds=min(values),
                        median_seconds=median,
                        maximum_seconds=max(values),
                        predictions_per_second=assets / median,
                        relative_to_reference_percent=100 * (median - reference) / reference,
                        recovery_median_seconds=statistics.median(
                            r["recovery_seconds"] for r in rows
                        ),
                        persisted_median_bytes=statistics.median(
                            r["persisted_bytes"] for r in rows
                        ),
                        book_median_bytes=statistics.median(r["book_bytes"] for r in rows),
                    )
                )
    return result


def _models(source):
    import torch

    from mars_titan.models.quantile_head import QUANTILE_HEAD
    from mars_titan.models.titans.episodic_readout import EpisodicReadout, EpisodicReadoutConfig
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
    from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

    result = {}
    for family in FAMILIES:
        config = FinancialConfig(source["spec"], variant="mac_online", hidden_size=32, seed=42)
        if family == "mechanism":
            config = replace(config, head=QUANTILE_HEAD)
        model = FinancialPredictor(config, device="cpu", dtype=torch.float64).eval()
        reader = EpisodicReadout(
            EpisodicReadoutConfig(
                source["codec"].fingerprint(), hidden_size=32, neighbors=4, seed=42
            ),
            dtype=torch.float64,
            device="cpu",
        ).eval()
        model.requires_grad_(False)
        reader.requires_grad_(False)
        constructor = FrozenFinancialConsumer if family == "scalar" else FrozenMechanismConsumer
        result[family] = constructor(model, readout=reader)
    return result


def _common(run, consumer, predictions, family):
    import torch

    from benchmarks.regional_decision_cuda import proof

    state = run.fast_state()
    flows = tuple(sorted(state["references"]))
    fast = consumer.predictor.export_state(run._fast_store.gather(state, flows))
    episodes = sorted(
        (r["sample_id"], r["label"], r["issued_prediction"], r["error"])
        for r in run.retained_episodes()
    )
    data = dict(
        points=sorted((p.asset, p.decision_at, p.value) for p in predictions),
        fast=fast,
        episodes=episodes,
    )
    if family == "mechanism":
        generation = run._bundle(run.snapshot()["state"])["generation"]
        event = run.mechanism_event(generation)["payload"]
        data.update(
            quantiles=[(r["flow_id"], r["raw_quantiles"], r["quantiles"]) for r in event["rows"]],
            auxiliary=run.mechanism_state(),
        )
    return proof(torch, data)[0]


def _case(native, source, consumer, directory, config, family):
    import torch
    from test_failure_session import book
    from test_financial_session import moment
    from test_financial_session_associative import label

    from mars_titan.memory.financial_session import FinancialPhase, FinancialSession
    from mars_titan.memory.retention_bank import RetentionConfig

    settings = dict(
        native=native,
        consumer=consumer,
        codec=source["codec"],
        prefixes=source["prefixes"],
        retention=RetentionConfig(capacity=64, frontier=2, new_candidates=4),
        phase=FinancialPhase("validation", moment(125), moment(125), moment(200), moment(201)),
        world="failure_cost_fixture",
        fold="0",
        admission="m1",
        block_rows=64,
        failure_memory=config,
    )
    run, pending = None, []
    rng = torch.get_rng_state().clone()
    try:
        run = FinancialSession(directory, **settings)
        for at in (125, 126, 127):
            labels = [native.Feedback(p.id, 0, moment(at), label(p.asset, at)) for p in pending]
            pending = run.step(source["batches"][at], labels).predictions
        snapshot = run.snapshot()
        run.close()
        run = FinancialSession(directory, **settings, resume=True)
        assert run.snapshot() == snapshot
        labels = [native.Feedback(p.id, 0, moment(128), label(p.asset, 128)) for p in pending]
        tick = time.perf_counter()
        result = run.step(source["batches"][128], labels)
        step = time.perf_counter() - tick
        common = _common(run, consumer, result.predictions, family)
        snapshot = run.snapshot()
        failure_state = None if config is None else book(run).export()
        logical = 0 if config is None else book(run).logical_bytes()
        run.close()
        tick = time.perf_counter()
        run = FinancialSession(directory, **settings, resume=True)
        recovery = time.perf_counter() - tick
        assert run.snapshot() == snapshot
        assert (None if config is None else book(run).export()) == failure_state
        persisted = sum(path.stat().st_size for path in directory.rglob("*") if path.is_file())
        labels = [
            native.Feedback(p.id, 0, moment(130), label(p.asset, 130)) for p in result.predictions
        ]
        following = run.step(source["batches"][130], labels)
        next_state = _common(run, consumer, following.predictions, family)
        consumer.verify(strong=True)
        assert torch.equal(rng, torch.get_rng_state())
        assert all(p.grad is None for p in consumer.predictor.parameters())
        return dict(
            step_seconds=step,
            recovery_seconds=recovery,
            persisted_bytes=persisted,
            book_bytes=logical,
            common_state_sha256=common,
            next_state_sha256=next_state,
        )
    finally:
        if run is not None:
            run.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--native", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    protocol = study_protocol(args.sizes, args.warmup, args.repeats)
    if not args.run:
        print(json.dumps(protocol, ensure_ascii=False))
        return
    root, out = args.repository.resolve(), args.output
    if (
        out is None
        or out.exists()
        or args.native is None
        or not args.native.is_file()
        or not 30 <= args.seconds <= 3600
    ):
        raise ValueError("La salida debe ser nueva y el presupuesto debe estar acotado")
    for part in ("src", ".", "tests/memory", "tests/models/titans"):
        sys.path.insert(0, str(root / part))
    import pyarrow as pa
    import torch
    from test_failure_session import policy
    from test_financial_session_controls import flow_source, no_target_estimation

    from benchmarks.regional_decision_cuda import proof
    from mars_titan.data.cohort_files import safe_destination
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.hardware.platform_identity import cpu_name
    from mars_titan.memory.native_backend import load_native
    from mars_titan.training.learning_hold import (
        hold_path,
        install_optimizer_guard,
        learning_blocked,
    )

    safe_destination(out)
    out = out.resolve()
    if out.is_relative_to(root) or not learning_blocked() or torch.cuda.is_initialized():
        raise ValueError("La medida CPU necesita salida externa y aprendizaje bloqueado")
    native = load_native(args.native)
    hold, attempts = hold_path(), []
    before_hold = sha256(hold)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    out.mkdir(parents=True)
    report = dict(
        schema_version=1,
        kind="failure_durable_session_technical_cost",
        status="running",
        measured_at_utc=datetime.now(UTC).isoformat(),
        protocol=protocol,
        hold_sha256_before=before_hold,
        optimizer_attempts=attempts,
        cases=[],
        fixtures=[],
        hardware=dict(cpu=cpu_name(), threads=1, gpu_used=False, vram_peak_bytes=None),
        software=dict(
            python=platform.python_version(), torch=torch.__version__, arrow=pa.__version__
        ),
        native_sha256=sha256(args.native),
        source_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            path: sha256(root / path)
            for path in (
                "benchmarks/failure_session_cost.py",
                "tests/memory/test_financial_session_controls.py",
                "tests/memory/test_failure_session.py",
                "src/mars_titan/memory/financial_session.py",
                "src/mars_titan/memory/failure_actions.py",
                "src/mars_titan/memory/failure_session.py",
                "src/mars_titan/memory/failure_producer.py",
                "src/mars_titan/memory/failure_ledger.py",
                "src/mars_titan/memory/failure_ledger_state.py",
            )
        },
        load_start=os.getloadavg(),
        limits=[
            "Fixtures manuales con modalidades compactas y un indicador macro, sin aprendizaje.",
            "No mide error predictivo, energía, transferencias GPU ni el corpus completo.",
            "El pico RSS incluye todos los casos, fixtures y comprobaciones del proceso.",
            "Tres repeticiones no estiman colas fiables. No se enfría la caché del disco.",
            "Solo se retiran casos técnicos terminados propios. Los fallidos se conservan.",
        ],
    )
    atomic_json(out / "protocol.json", protocol)

    def refuse(reason):
        attempts.append(reason)
        raise RuntimeError(reason)

    def alarm(*_):
        raise TimeoutError("El coste técnico agotó el presupuesto de proceso")

    guard = install_optimizer_guard(refuse, path=hold)
    assert guard is not None
    old_signal = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    started = time.perf_counter()
    try:
        with contextmanager(no_target_estimation.__wrapped__)():
            for assets in args.sizes:
                source = flow_source(
                    out / f"fixture-{assets}", flows=tuple(f"US/T{i:04d}" for i in range(assets))
                )
                models = _models(source)
                report["fixtures"].append(
                    dict(
                        assets=assets,
                        specification=source["spec"].identity(),
                        models={family: model.identity() for family, model in models.items()},
                    )
                )
                model_hashes = {
                    family: proof(torch, model.predictor.state_dict())[0]
                    for family, model in models.items()
                }
                controls = [(family, enabled) for family in FAMILIES for enabled in (False, True)]
                for repetition in range(args.warmup + args.repeats):
                    shift = repetition % len(controls)
                    for family, enabled in controls[shift:] + controls[:shift]:
                        consumer = models[family]
                        config = policy(source, consumer, out)
                        config = replace(
                            config,
                            ledger=replace(
                                config.ledger,
                                capacity=assets,
                                max_pending=4 * assets,
                                max_flows=assets,
                            ),
                        )
                        if family == "mechanism":
                            config = replace(config, action_source="mechanism_trace_v1")
                        directory = out / f"case-{assets}-{family}-{int(enabled)}-{repetition}"
                        record = _case(
                            native, source, consumer, directory, config if enabled else None, family
                        )
                        record.update(
                            assets=assets,
                            family=family,
                            enabled=enabled,
                            repetition=repetition,
                            warmup=repetition < args.warmup,
                            data_source_sha256=source["spec"].source_sha256,
                            failure_configuration=config.identity() if enabled else None,
                        )
                        shutil.rmtree(directory)
                        record["case_directory_retained"] = False
                        report["cases"].append(record)
                        atomic_json(out / "measurement.json", report)
                        print(
                            json.dumps(
                                {
                                    k: record[k]
                                    for k in ("assets", "family", "enabled", "step_seconds")
                                }
                            ),
                            flush=True,
                        )
                    if repetition >= args.warmup:
                        summarize(report["cases"])
                assert all(
                    proof(torch, model.predictor.state_dict())[0] == model_hashes[family]
                    for family, model in models.items()
                )
                del source, models
                gc.collect()
        assert not attempts and sha256(hold) == before_hold
        report.update(status="completed", summary=summarize(report["cases"]))
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        guard.remove()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old_signal)
        report.update(
            hold_sha256_after=sha256(hold),
            elapsed_seconds=time.perf_counter() - started,
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            load_end=os.getloadavg(),
        )
        atomic_json(out / "measurement.json", report)


if __name__ == "__main__":
    main()
