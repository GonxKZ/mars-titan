"""Coste técnico de cargar parámetros elegidos, sin inferencia ni entrenamiento."""

import hashlib
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import mars_titan
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.training import spaced_selected
from mars_titan.training.spaced_selected import load_selected_reader
from tests.memory.spaced_selected_fixture import selected_fixture

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]
from test_financial_session_controls import (  # noqa: E402
    flow_source,
    no_target_estimation,
)


def main():
    assert Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    assert not torch.cuda.is_initialized()

    def reject_optimizer(*_):
        raise RuntimeError("La medida no admite pasos de optimizador")

    handle = register_optimizer_step_pre_hook(reject_optimizer)
    guard = no_target_estimation.__wrapped__()
    next(guard)
    started = time.perf_counter()
    rows = []
    report = dict(
        kind="technical_selected_reader_load_cost",
        status="running",
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            device="cpu",
            dtype="float64",
            cpu_threads=2,
        ),
        hardware=dict(cpu=cpu_name(), platform=platform.platform()),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        implementation_sha256=hashlib.sha256(
            Path(spaced_selected.__file__).read_bytes()
        ).hexdigest(),
        fixture_sha256=hashlib.sha256(
            (ROOT / "tests/memory/spaced_selected_fixture.py").read_bytes()
        ).hexdigest(),
        measured_scope="metadata_checkpoints_model_construction_freezing_and_revalidation",
        excludes=[
            "imports",
            "fixture_preparation",
            "forecasting",
            "dataset_auditing",
            "GPU",
            "energy",
        ],
        fixture_prediction_artifacts=0,
        quantile_head=True,
        warmup=2,
        measured_repetitions=10,
        samples=rows,
    )
    try:
        source = flow_source(out / "data")
        for hidden in (32, 64, 128):
            fixture = selected_fixture(
                out / f"weights-{hidden}", source, hidden_size=hidden, quantiles=True
            )
            expected = None
            descriptor = fixture["selected"]["readout"]
            checkpoint = fixture["anchor"] / descriptor["path"]
            parent = fixture["anchor"].parent / "parent"
            parent_report = json.loads((parent / "run.json").read_text())
            checkpoint_bytes = (
                checkpoint.stat().st_size
                + (parent / parent_report["checkpoint"]["path"]).stat().st_size
            )
            for repetition in range(12):
                before = time.perf_counter()
                loaded = load_selected_reader(
                    fixture["anchor"], fixture["view"], source["spec"], device="cpu"
                )
                elapsed = time.perf_counter() - before
                current = loaded.consumer.model_id
                if expected is None:
                    expected = current
                assert current == expected
                rows.append(
                    dict(
                        hidden_size=hidden,
                        repetition=repetition,
                        warmup=repetition < 2,
                        seconds=elapsed,
                        selected_checkpoint_bytes=checkpoint_bytes,
                        consumer_id=current,
                    )
                )
                del loaded
            atomic_json(out / "cost.json", report)
        report["summaries"] = [
            dict(
                hidden_size=hidden,
                median_seconds=statistics.median(values),
                min_seconds=min(values),
                max_seconds=max(values),
                samples=len(values),
            )
            for hidden in (32, 64, 128)
            for values in [
                [r["seconds"] for r in rows if r["hidden_size"] == hidden and not r["warmup"]]
            ]
        ]
        report.update(
            status="completed",
            seconds_without_imports=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            cuda_initialized=torch.cuda.is_initialized(),
        )
        atomic_json(out / "cost.json", report)
        print(json.dumps(dict(status=report["status"], summaries=report["summaries"])))
    finally:
        handle.remove()
        guard.close()


if __name__ == "__main__":
    main()
