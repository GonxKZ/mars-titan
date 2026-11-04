"""Coste completo de una fase durable regional con fixtures y parámetros compartidos fijos."""

import hashlib
import json
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import mars_titan
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.native_backend import load_native

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]
from test_financial_session import moment  # noqa: E402
from test_financial_session_controls import flow_source, no_target_estimation  # noqa: E402
from test_spaced_regional_session import regional_consumer  # noqa: E402
from test_spaced_session import feedback, open_session  # noqa: E402


def file_bytes(output):
    paths = subprocess.check_output(["rg", "--files", "--hidden", "--no-ignore", "-0", str(output)])
    return sum(Path(path.decode()).stat().st_size for path in paths.split(b"\0") if path)


def sample(native, source, consumer, output, *, enabled, assets):
    policy = "reservoir" if enabled else None
    capacity, block = 2 * assets, min(16, assets)
    emitted, diagnostics = [], []
    started = time.perf_counter()
    with open_session(
        native, source, consumer, output, policy=policy, capacity=capacity, block_rows=block
    ) as session:
        previous = []
        for at in (125, 126):
            event = session.step(source["batches"][at], feedback(native, previous, at))
            emitted.extend((p.asset, p.decision_at, p.value) for p in event.predictions)
            previous = event.predictions
    with open_session(
        native,
        source,
        consumer,
        output,
        policy=policy,
        capacity=capacity,
        block_rows=block,
        resume=True,
    ) as session:
        for at in (127, 128):
            event = session.step(source["batches"][at], feedback(native, previous, at))
            emitted.extend((p.asset, p.decision_at, p.value) for p in event.predictions)
            previous = event.predictions
            bundle = session._bundle(session.snapshot()["state"])
            control = session._read(bundle["control"], "control")
            diagnostics.append(control.get("spaced_regional"))
        session.step([], [], kind="settlement", cutoff=moment(201), close_phase=True)
        snapshot = session.snapshot()
        assert snapshot["phase_closed"] is True
    elapsed = time.perf_counter() - started
    assert len(emitted) == assets * 4
    return dict(
        seconds=elapsed,
        emitted_rows=len(emitted),
        emitted_sha256=hashlib.sha256(json.dumps(emitted, sort_keys=True).encode()).hexdigest(),
        persistent_bytes=file_bytes(output),
        diagnostic_work=diagnostics,
    )


def main():
    if not Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src"):
        raise RuntimeError("El benchmark necesita las fuentes propias con PYTHONPATH=src:.")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    native = load_native()
    guard = no_target_estimation.__wrapped__()
    next(guard)
    attempts = 0

    def forbidden(*_):
        nonlocal attempts
        attempts += 1
        raise RuntimeError("La medida durable no permite pasos de optimizador")

    hook = register_optimizer_step_pre_hook(forbidden)
    attention = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)
    samples = []
    started = time.perf_counter()
    report = dict(
        kind="technical_durable_regional_spaced_session_cost",
        status="running",
        runtime=dict(
            torch=str(torch.__version__),
            python=platform.python_version(),
            device="cpu",
            dtype="torch.float64",
            cpu_threads=2,
            attention_fastpath=False,
            native_sha256=native.binary_sha256,
        ),
        hardware=dict(cpu=cpu_name(), platform=platform.platform()),
        protocol=dict(
            assets=[4, 64],
            readers=["ring", "tree"],
            warmup=1,
            repetitions=3,
            cohort_decisions=[125, 126, 127, 128],
            reopen_after=126,
            close_at=201,
            total_budget_seconds=600,
            memory_capacity="twice_assets",
        ),
        samples=samples,
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(
            (ROOT / "tests/memory/test_spaced_regional_session.py").read_bytes()
        ).hexdigest(),
        limits=[
            "Fixtures con etiquetas manuales y dimensiones 5/2/1/3/3. No corpus real.",
            "Incluye creación de sesión, cuatro cohortes, cierre, "
            "reapertura y auditorías durables.",
            "Incluye codificación del predictor y consulta. La creación del fixture queda fuera.",
            "Los parámetros compartidos se cargan una vez por caso, fuera del cronómetro.",
            "El inventario de bytes persistentes se calcula después del tiempo por fase.",
            "No incluye codificadores externos de noticias o imágenes, GPU ni transferencias.",
            "Tres muestras por modo no permiten caracterizar p95/p99. No se mide energía.",
        ],
    )
    try:
        for assets in (4, 64):
            source = flow_source(
                out / f"input-{assets}", flows=tuple(f"US/T{i:04d}" for i in range(assets))
            )
            assert source["label_origin"] == "manual_fixture_no_estimation"
            for reader in ("ring", "tree"):
                consumer = regional_consumer(source, reader)
                for repetition in range(4):
                    pair = []
                    for enabled in (False, True) if repetition % 2 == 0 else (True, False):
                        if time.perf_counter() - started >= 600:
                            report.update(status="budget_exhausted")
                            atomic_json(out / "cost.json", report)
                            raise RuntimeError("La medida ha agotado sus 600 segundos")
                        result = sample(
                            native,
                            source,
                            consumer,
                            out / f"run-{assets}-{reader}-{repetition}-{int(enabled)}",
                            enabled=enabled,
                            assets=assets,
                        )
                        entry = dict(
                            result,
                            assets=assets,
                            reader=reader,
                            enabled=enabled,
                            repetition=repetition,
                            warmup=repetition == 0,
                        )
                        samples.append(entry)
                        pair.append(entry)
                    assert pair[0]["emitted_sha256"] == pair[1]["emitted_sha256"]
                    atomic_json(out / "cost.json", report)
        report["summaries"] = [
            dict(
                assets=assets,
                reader=reader,
                enabled=enabled,
                median_seconds=statistics.median(values),
                min_seconds=min(values),
                max_seconds=max(values),
                repetitions=len(values),
            )
            for assets in (4, 64)
            for reader in ("ring", "tree")
            for enabled in (False, True)
            for values in [
                [
                    v["seconds"]
                    for v in samples
                    if v["assets"] == assets
                    and v["reader"] == reader
                    and v["enabled"] is enabled
                    and not v["warmup"]
                ]
            ]
        ]
        report.update(
            status="completed",
            seconds_without_imports=time.perf_counter() - started,
            process_peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            optimizer_attempts=attempts,
            cuda_initialized=torch.cuda.is_initialized(),
        )
        assert attempts == 0 and report["cuda_initialized"] is False
        atomic_json(out / "cost.json", report)
        print(json.dumps(dict(status=report["status"], summaries=report["summaries"])))
    finally:
        torch.backends.mha.set_fastpath_enabled(attention)
        hook.remove()
        guard.close()


if __name__ == "__main__":
    main()
