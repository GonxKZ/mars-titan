"""Coste CPU de N2 en la sesión nativa con fixtures manuales y sin aprendizaje.

Sin --run solo imprime el protocolo. La decisión medida consulta un banco no vacío.
La maduración ocurre en el prefijo, fuera del intervalo de step. La recuperación
mantiene el modelo residente. No mide calidad, codificación original, energía ni coste CUDA.
"""

import argparse
import json
import math
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
from datetime import UTC, datetime
from pathlib import Path

VARIANTS = ("reference", "zero_gain", "observable", "past_prefix", "constant")
COMPOSITIONS = ("n2", "e2_reservoir", "e2_utility", "mechanisms", "collection")
SETTINGS = (
    (None, 0.0),
    ("observable", 0.0),
    ("observable", 1.0),
    ("shuffled_past_prefix", 1.0),
    ("constant", 1.0),
)


def study_protocol(sizes, warmup, repeats, *, compositions=False):
    if (
        type(compositions) is not bool
        or type(sizes) is not list
        or not sizes
        or any(type(n) is not int or n not in (4, 64, 256) for n in sizes)
        or len(sizes) != len(set(sizes))
        or type(warmup) is not int
        or not 0 <= warmup <= 3
        or type(repeats) is not int
        or not 1 <= repeats <= 10
    ):
        raise ValueError("El protocolo N2 necesita tamaños únicos y repeticiones acotadas")
    variants = COMPOSITIONS if compositions else VARIANTS
    result = dict(
        sizes=sizes,
        variants=list(variants),
        warmup=warmup,
        repeats=repeats,
        total_cases=len(sizes) * len(variants) * (warmup + repeats),
        measured_cases=len(sizes) * len(variants) * repeats,
        device="cpu",
        dtype="torch.float64",
        threads=1,
        hidden_size=32,
        block_rows=64,
        episodic_capacity=64,
        memory_reads=1,
        admission="m1",
        min_assets=20,
        min_returns=42,
        recent_returns=21,
        fixture="manual_supervision_compact_modalities_one_macro_indicator",
        prefix_events=[125, 126],
        measured_event=127,
        following_event=128,
        parameter_fixture="reference_or_fixed_gain_0_or_1_no_fitting",
    )
    if compositions:
        result.update(
            head="quantile_head_v1",
            parameter_fixture="same_quantile_predictor_and_reader_fixed_gain_1_no_fitting",
            parity_variants=[name for name in COMPOSITIONS if name != "e2_utility"],
            comparison="integration_and_retention_with_the_same_quantile_head",
        )
    return result


def summarize(cases, *, compositions=False):
    result = []
    variants = COMPOSITIONS if compositions else VARIANTS
    parity = (
        [name for name in COMPOSITIONS if name != "e2_utility"]
        if compositions
        else ("reference", "zero_gain")
    )
    for size in sorted({r["assets"] for r in cases}):
        group = [r for r in cases if r["assets"] == size and not r["warmup"]]
        if {r["variant"] for r in group} != set(variants):
            raise ValueError("Faltan variantes del coste N2")
        if len({r["source_sha256"] for r in group}) != 1:
            raise ValueError("Las variantes no comparten fuente")
        repetitions = None
        for variant in variants:
            rows = [r for r in group if r["variant"] == variant]
            if any(
                type(r[k]) not in (int, float) or not math.isfinite(r[k]) or r[k] <= 0
                for r in rows
                for k in ("step_seconds", "recovery_seconds")
            ):
                raise ValueError("Los tiempos deben ser finitos y positivos")
            if any(len({r[k] for r in rows}) != 1 for k in ("common_sha256", "next_sha256")):
                raise ValueError("La misma variante no reproduce su estado")
            current = [r["repetition"] for r in rows]
            if len(current) != len(set(current)) or (
                repetitions is not None and set(current) != repetitions
            ):
                raise ValueError("Las repeticiones no están emparejadas")
            repetitions = set(current)
            if variant in parity:
                pair = [r for r in group if r["variant"] in parity]
                if any(len({r[k] for r in pair}) != 1 for k in ("common_sha256", "next_sha256")):
                    raise ValueError("El control equivalente perdió paridad con la referencia")
            values = [r["step_seconds"] for r in rows]
            result.append(
                dict(
                    assets=size,
                    variant=variant,
                    samples=len(values),
                    median_seconds=statistics.median(values),
                    minimum_seconds=min(values),
                    maximum_seconds=max(values),
                    recovery_median_seconds=statistics.median(r["recovery_seconds"] for r in rows),
                    predictions_per_second=size / statistics.median(values),
                )
            )
    return result


def _common(session, predictions):
    import torch

    from benchmarks.regional_decision_cuda import proof
    from mars_titan.models.titans.state import mac_tensors

    bundle = session._bundle(session.snapshot()["state"])
    fast = session.fast_state()
    flows = tuple(sorted(fast["references"]))
    state = session._fast_store.gather(fast, flows)
    bank, episodes = session._bank(bundle["bank"])
    records = []
    for record in bank.records():
        row = episodes[record.id]
        records.append(
            (
                row["flow_id"],
                row["sample_id"],
                row["issued_prediction"],
                row["error"],
                record.key,
                record.value,
                record.label,
                record.decision_at,
                record.available_at,
                record.maturity_at,
            )
        )
    data = dict(
        points=sorted((p.asset, p.value) for p in predictions),
        flows=flows,
        tensors=tuple(mac_tensors(state.mac)),
        observed=state.observed_steps,
        episodes=sorted(records, key=lambda r: (r[0], r[1])),
    )
    return proof(torch, data)[0]


def _composition_options(name):
    from mars_titan.memory.native_mechanism_collection import MechanismCollectionConfig
    from mars_titan.memory.spaced_retention import SpacedRetentionConfig

    if name not in COMPOSITIONS:
        raise ValueError("La composición N2 no está registrada")
    if name.startswith("e2_"):
        return dict(
            retention=SpacedRetentionConfig(
                policy=name.removeprefix("e2_"), capacity=64, frontier=2, seed=73
            )
        )
    return dict(mechanism_collection=MechanismCollectionConfig()) if name == "collection" else {}


def _models(source, *, compositions):
    from tests.memory.dendritic_fixtures import engine

    if not compositions:
        return {
            name: engine(source, mode, gain=gain, min_assets=20)
            for name, (mode, gain) in zip(VARIANTS, SETTINGS, strict=True)
        }
    from mars_titan.models.quantile_head import QUANTILE_HEAD
    from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer

    base = engine(source, "observable", gain=1.0, min_assets=20, head=QUANTILE_HEAD)
    mechanisms = FrozenMechanismConsumer(base.predictor, readout=base.readout)
    return dict(
        n2=base, e2_reservoir=base, e2_utility=base, mechanisms=mechanisms, collection=mechanisms
    )


def _case(native, source, consumer, directory, *, composition=None):
    from tests.memory.dendritic_fixtures import run
    from tests.memory.test_financial_session import moment

    settings = dict(block_rows=64, capacity=64)
    if composition is not None:
        settings.update(_composition_options(composition))
    session = None
    try:
        session = run(native, directory, source, consumer, **settings)
        first = session.step(source["batches"][125], [])
        session.step(
            source["batches"][126],
            [native.Feedback(p.id, 0, moment(126), 0.125) for p in first.predictions],
        )
        snapshot = session.snapshot()
        session.close()
        session = run(native, directory, source, consumer, resume=True, **settings)
        assert session.snapshot() == snapshot
        bank_count = len(session.retained_episodes())
        assert bank_count > 0
        start = time.perf_counter()
        third = session.step(source["batches"][127], [])
        elapsed = time.perf_counter() - start
        control = session._read(session._bundle(session.snapshot()["state"])["control"], "control")
        if "snapshot_bytes" in control:
            assert control["snapshot_bytes"] > 0
        else:
            assert control["snapshot_sha256"] is not None
        common = _common(session, third.predictions)
        context = control.get("dendritic_context")
        context_bytes = 0 if context is None else len(json.dumps(context, sort_keys=True).encode())
        snapshot = session.snapshot()
        session.close()
        start = time.perf_counter()
        session = run(native, directory, source, consumer, resume=True, **settings)
        recovery = time.perf_counter() - start
        assert session.snapshot() == snapshot
        following = session.step(source["batches"][128], [])
        next_state = _common(session, following.predictions)
        consumer.verify(strong=True)
        return dict(
            consumer_id=consumer.model_id,
            contract_id=session.contract_id,
            binding_kind=session._binding.kind,
            spaced_policy=getattr(getattr(session._binding, "spaced_config", None), "policy", None),
            collection_enabled=getattr(session._binding, "collections", None) is not None,
            bank_count_before_step=bank_count,
            step_seconds=elapsed,
            recovery_seconds=recovery,
            common_sha256=common,
            next_sha256=next_state,
            context_record_bytes=context_bytes,
            persisted_bytes=sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()),
        )
    finally:
        if session is not None:
            session.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--sizes", type=int, nargs="+", default=[64, 256])
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--compositions", action="store_true")
    args = parser.parse_args()
    protocol = study_protocol(args.sizes, args.warmup, args.repeats, compositions=args.compositions)
    variants = tuple(protocol["variants"])
    if not args.run:
        print(json.dumps(protocol, ensure_ascii=False))
        return
    root = args.repository.resolve()
    out = args.output
    if (
        out is None
        or out.exists()
        or args.native is None
        or not args.native.is_file()
        or not 30 <= args.seconds <= 1800
        or out.resolve().is_relative_to(root)
    ):
        raise ValueError("La salida debe ser nueva, externa y con presupuesto y enlace válidos")
    for part in ("src", ".", "tests/memory", "tests/models/titans"):
        sys.path.insert(0, str(root / part))
    import pyarrow as pa
    import torch

    from mars_titan.data.cohort_files import safe_destination
    from mars_titan.data.storage import atomic_json, sha256
    from mars_titan.hardware.platform_identity import cpu_name
    from mars_titan.memory.native_backend import load_native
    from mars_titan.training.learning_hold import (
        hold_path,
        install_optimizer_guard,
        learning_blocked,
    )
    from tests.memory.dendritic_fixtures import build_source
    from tests.memory.test_financial_session_controls import no_target_estimation

    if not learning_blocked() or torch.cuda.is_initialized():
        raise ValueError("La medida CPU necesita aprendizaje bloqueado y CUDA sin inicializar")
    safe_destination(out)
    out.mkdir(parents=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    pa.set_cpu_count(1)
    pa.set_io_thread_count(1)
    native = load_native(args.native)
    hold = hold_path()
    before = sha256(hold)
    attempts = []
    report = dict(
        schema_version=1,
        kind="dendritic_composition_native_session_technical_cost"
        if args.compositions
        else "dendritic_native_session_technical_cost",
        status="running",
        measured_at_utc=datetime.now(UTC).isoformat(),
        protocol=protocol,
        cases=[],
        fixtures=[],
        hardware=dict(cpu=cpu_name(), gpu_used=False, threads=1, vram_peak_bytes=None),
        software=dict(
            python=platform.python_version(), torch=torch.__version__, arrow=pa.__version__
        ),
        native_sha256=sha256(args.native),
        hold_sha256_before=before,
        optimizer_attempts=attempts,
        source_commit=subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        source_sha256={
            p: sha256(root / p)
            for p in (
                "benchmarks/dendritic_session_cost.py",
                "tests/memory/dendritic_fixtures.py",
                "tests/memory/test_financial_session_controls.py",
                "src/mars_titan/memory/financial_session.py",
                "src/mars_titan/memory/financial_consumers.py",
                "src/mars_titan/memory/dendritic_session.py",
                "src/mars_titan/models/titans/financial.py",
                "src/mars_titan/models/titans/frozen_financial.py",
                "src/mars_titan/models/titans/dendritic_context.py",
                "src/mars_titan/models/titans/dendritic_regime.py",
                "src/mars_titan/models/titans/active_dendrites.py",
                "src/mars_titan/models/titans/mechanism_consumer.py",
                "src/mars_titan/memory/financial_mechanisms.py",
                "src/mars_titan/memory/mechanism_collection.py",
                "src/mars_titan/memory/spaced_session.py",
                "src/mars_titan/memory/spaced_retention.py",
            )
        },
        load_start=os.getloadavg(),
        limits=[
            "Fixtures manuales compactos y parámetros sin entrenar. No mide calidad predictiva.",
            "La recuperación mantiene el modelo congelado residente. No mide arranque en frío.",
            "El pico RSS es global. No mide memoria por variante, energía o transferencias CUDA.",
            "Tres repeticiones son descriptivas. Hay otras aplicaciones activas.",
            "La prueba del banco compara procedencia y contenidos y excluye IDs del runtime.",
        ],
    )

    def refuse(reason):
        attempts.append(reason)
        raise RuntimeError(reason)

    def deadline(*_):
        raise TimeoutError("El coste N2 agotó su presupuesto")

    guard = install_optimizer_guard(refuse, path=hold)
    assert guard is not None
    prior = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, args.seconds)
    started = time.perf_counter()
    atomic_json(out / "protocol.json", protocol)
    try:
        with contextmanager(no_target_estimation.__wrapped__)():
            for assets in args.sizes:
                source = build_source(
                    out / f"fixture-{assets}", flows=tuple(f"US/T{i:04d}" for i in range(assets))
                )
                models = _models(source, compositions=args.compositions)
                report["fixtures"].append(
                    dict(
                        assets=assets,
                        inputs=source["spec"].identity(),
                        models={k: v.identity() for k, v in models.items()},
                        model_ids={k: v.model_id for k, v in models.items()},
                    )
                )
                rng = torch.get_rng_state().clone()
                for repetition in range(args.warmup + args.repeats):
                    shift = repetition % len(variants)
                    order = variants[shift:] + variants[:shift]
                    for variant in order:
                        directory = out / f"case-{assets}-{variant}-{repetition}"
                        options = dict(composition=variant) if args.compositions else {}
                        record = _case(native, source, models[variant], directory, **options)
                        assert record["consumer_id"] == models[variant].model_id
                        record.update(
                            assets=assets,
                            variant=variant,
                            repetition=repetition,
                            warmup=repetition < args.warmup,
                            source_sha256=source["spec"].source_sha256,
                        )
                        shutil.rmtree(directory)
                        record["case_directory_retained"] = False
                        report["cases"].append(record)
                        atomic_json(out / "measurement.json", report)
                        print(
                            json.dumps(
                                {k: record[k] for k in ("assets", "variant", "step_seconds")}
                            ),
                            flush=True,
                        )
                    if repetition >= args.warmup:
                        summarize(report["cases"], compositions=args.compositions)
                assert torch.equal(rng, torch.get_rng_state())
                for consumer in models.values():
                    consumer.verify(strong=True)
        assert not attempts and sha256(hold) == before
        report.update(
            status="completed", summary=summarize(report["cases"], compositions=args.compositions)
        )
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        guard.remove()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prior)
        report.update(
            hold_sha256_after=sha256(hold),
            elapsed_seconds=time.perf_counter() - started,
            peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            load_end=os.getloadavg(),
        )
        atomic_json(out / "measurement.json", report)


if __name__ == "__main__":
    main()
