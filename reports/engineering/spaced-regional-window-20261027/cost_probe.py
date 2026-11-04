"""Coste del controlador E2 regional completo con ventanas de fixture, sin aprendizaje."""

import hashlib
import json
import platform
import resource
import statistics
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

import mars_titan
from mars_titan.data.storage import atomic_json
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.native_backend import load_native
from mars_titan.training import spaced_inference, spaced_window
from tests.memory.spaced_window_fixture import study_fixture

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "tests/memory"), str(ROOT / "tests/models/titans")]
from test_financial_session_controls import no_target_estimation  # noqa: E402
from test_spaced_regional_selected import components  # noqa: E402


def signature(output, report):
    values = []
    for name, item in sorted(report["predictions"].items()):
        table = pq.read_table(output / item["path"])
        values.append((name, table.to_pydict()))
    return hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()


def main():
    assert Path(mars_titan.__file__).resolve().is_relative_to(ROOT / "src")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    native = load_native()
    assert not torch.cuda.is_initialized()
    guard = no_target_estimation.__wrapped__()
    next(guard)
    optimizer_attempts = 0

    def reject_optimizer(*_):
        nonlocal optimizer_attempts
        optimizer_attempts += 1
        raise RuntimeError("El benchmark técnico no admite optimizadores")

    handle = register_optimizer_step_pre_hook(reject_optimizer)
    rows, started = [], time.perf_counter()
    result = dict(
        kind="technical_regional_frozen_study_controller_cost",
        status="running",
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            device="cpu",
            dtype="float64",
            cpu_threads=2,
            native_sha256=native.binary_sha256,
        ),
        hardware=dict(cpu=cpu_name(), platform=platform.platform()),
        probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        controller_sha256=hashlib.sha256(Path(spaced_window.__file__).read_bytes()).hexdigest(),
        fixture_sha256=hashlib.sha256(
            (ROOT / "tests/memory/spaced_window_fixture.py").read_bytes()
        ).hexdigest(),
        total_budget_seconds=600,
        warmup=1,
        measured_repetitions=3,
        measured_scope="controller_metadata_selected_weights_sources_three_phases_predictions_sessions_evidence_and_audits",
        excludes=["imports", "fixture_preparation", "postrun_comparison", "GPU", "energy"],
        samples=rows,
    )
    try:
        for assets in (4, 64):
            fixture = study_fixture(
                out / f"fixture-{assets}",
                assets=assets,
                rich=True,
                components=components("tree"),
                reader_contract="regional",
            )
            assert fixture["dataset"].manifest["technical_fixture"] is True

            def fixture_only(*_, current=fixture):
                if not current["dataset"].manifest["technical_fixture"]:
                    raise RuntimeError("Solo se permiten los datos creados por este fixture")

            with pytest.MonkeyPatch.context() as scope:
                scope.setattr(spaced_window, "require_learning_allowed", fixture_only)
                scope.setattr(spaced_inference, "require_learning_allowed", fixture_only)
                arms = [
                    "baseline",
                    "reservoir",
                    "utility",
                    "permuted_utility",
                    "uniform_frontier",
                    "recent",
                ]
                for repetition in range(4):
                    current = []
                    for arm in arms if repetition % 2 == 0 else reversed(arms):
                        output = out / f"run-{assets}-{repetition}-{arm}"
                        if time.perf_counter() - started >= 600:
                            result["status"] = "budget_exhausted"
                            atomic_json(out / "cost.json", result)
                            raise RuntimeError("La medida regional ha agotado sus 600 segundos")
                        before = time.perf_counter()
                        report = spaced_window.run_spaced_window(
                            fixture["view"],
                            fixture["anchor"],
                            fixture["source_manifest"],
                            fixture["study"],
                            arm=arm,
                            output=output,
                            device="cpu",
                            indices=fixture["indices"],
                        )
                        elapsed = time.perf_counter() - before
                        assert report["status"] == "completed"
                        entry = dict(
                            assets=assets,
                            arm=arm,
                            repetition=repetition,
                            warmup=repetition == 0,
                            seconds=elapsed,
                            samples=sum(v["rows"] for v in report["predictions"].values()),
                            predictions_sha256=signature(output, report),
                        )
                        rows.append(entry)
                        current.append(entry)
                    first = next(v for v in current if v["arm"] == "baseline")
                    second = next(v for v in current if v["arm"] == "reservoir")
                    assert first["predictions_sha256"] == second["predictions_sha256"]
                    assert len({v["samples"] for v in current}) == 1
                    atomic_json(out / "cost.json", result)
        result["summaries"] = [
            dict(
                assets=assets,
                arm=arm,
                median_seconds=statistics.median(values),
                min_seconds=min(values),
                max_seconds=max(values),
                repetitions=len(values),
            )
            for assets in (4, 64)
            for arm in [
                "baseline",
                "reservoir",
                "utility",
                "permuted_utility",
                "uniform_frontier",
                "recent",
            ]
            for values in [
                [
                    v["seconds"]
                    for v in rows
                    if v["assets"] == assets and v["arm"] == arm and not v["warmup"]
                ]
            ]
        ]
        result.update(
            status="completed",
            seconds_without_imports=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            optimizer_attempts=optimizer_attempts,
            cuda_initialized=torch.cuda.is_initialized(),
        )
        atomic_json(out / "cost.json", result)
        print(json.dumps(dict(status=result["status"], summaries=result["summaries"])))
    finally:
        handle.remove()
        guard.close()


if __name__ == "__main__":
    main()
