"""Medir el comparador E2 completo con informes manuales, sin modelos ni aprendizaje."""

import copy
import cProfile
import json
import platform
import pstats
import resource
import statistics
import sys
import time
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch.optim.optimizer import register_optimizer_step_pre_hook

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import spaced_comparison
from mars_titan.hardware.platform_identity import cpu_name
from mars_titan.memory.spaced_study import load_study
from mars_titan.training.spaced_window import _digest
from tests.evaluation.spaced_run_fixture import manual_runs, update_prediction
from tests.memory.spaced_analysis_fixture import ANALYSIS

ROOT = Path(__file__).resolve().parents[3]


def fixture(root, assets, quantiles):
    source = manual_runs(root, quantiles=quantiles, days=48)
    configuration = json.loads(source["study"].read_text())
    configuration["analysis"] = copy.deepcopy(ANALYSIS)
    atomic_json(source["study"], configuration)
    study, study_sha = load_study(source["study"])
    factor = assets // 4
    for path in source["reports"].values():
        record = json.loads(path.read_text())
        record["request"].update(study=study.identity(), study_sha256=study_sha)
        record["identity"]["request"] = copy.deepcopy(record["request"])
        for part in tuple(record["predictions"]):
            original = pq.read_table(path.parent / record["predictions"][part]["path"])
            flows = original["asset_id"].to_pylist()
            times = original["prediction_at"].cast(pa.int64()).to_pylist()
            chunks = []
            for replica in range(factor):
                current_flows = [f"{flow}_r{replica}" for flow in flows]
                identifiers = [
                    f"{flow}/{at}" for flow, at in zip(current_flows, times, strict=True)
                ]
                current = original.set_column(
                    original.column_names.index("asset_id"), "asset_id", pa.array(current_flows)
                )
                current = current.set_column(
                    current.column_names.index("sample_id"), "sample_id", pa.array(identifiers)
                )
                chunks.append(current)
            table = pa.concat_tables(chunks)
            update_prediction(path.parent, record, part, table)
            prediction = path.parent / record["predictions"][part]["path"]
            pq.write_table(table, prediction, row_group_size=4096, compression="zstd")
            record["predictions"][part].update(
                sha256=sha256(prediction), bytes=prediction.stat().st_size
            )
            record["identity"]["expected_rows"][part] = len(table)
        record["run_id"] = _digest(record["identity"])
        atomic_json(path, record)
    return source


def signature(report):
    return {key: report[key] for key in ("scores", "primary", "comparisons", "cohort_sha256")}


def profile_rows(profiler):
    entries = []
    for (filename, line, function), (primitive, calls, own, cumulative, _) in pstats.Stats(
        profiler
    ).stats.items():
        path = Path(filename)
        if path.is_relative_to(ROOT):
            filename = str(path.relative_to(ROOT))
        elif "site-packages/" in filename:
            filename = filename.split("site-packages/", 1)[1]
        else:
            filename = path.name
        entries.append(
            dict(
                file=filename,
                line=line,
                function=function,
                calls=calls,
                primitive_calls=primitive,
                own_seconds=own,
                cumulative_seconds=cumulative,
            )
        )
    return sorted(entries, key=lambda row: row["cumulative_seconds"], reverse=True)[:40]


def main():
    assert Path(spaced_comparison.__file__).resolve().is_relative_to(ROOT / "src")
    output = Path(sys.argv[1])
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    assert not torch.cuda.is_initialized()
    optimizer_attempts = 0

    def reject_optimizer(*_):
        nonlocal optimizer_attempts
        optimizer_attempts += 1
        raise RuntimeError("El coste técnico no admite pasos de optimizador")

    hook = register_optimizer_step_pre_hook(reject_optimizer)
    started, samples, inputs = time.perf_counter(), [], []
    result = dict(
        kind="technical_spaced_comparison_cost_without_models_or_learning",
        status="running",
        samples=samples,
        fixtures=inputs,
        budget_seconds=600,
        warmup=1,
        measured_repetitions=3,
        source_code={
            name: sha256(ROOT / name)
            for name in (
                "src/mars_titan/memory/spaced_study.py",
                "src/mars_titan/evaluation/spaced_sources.py",
                "src/mars_titan/evaluation/spaced_comparison.py",
                "src/mars_titan/evaluation/forecast_files.py",
                "src/mars_titan/evaluation/paired_comparisons.py",
                "tests/evaluation/spaced_run_fixture.py",
            )
        },
        probe_sha256=sha256(Path(__file__)),
        hardware=dict(cpu=cpu_name(), platform=platform.platform()),
        runtime=dict(
            python=platform.python_version(),
            torch=str(torch.__version__),
            numpy=np.__version__,
            pyarrow=pa.__version__,
            duckdb=duckdb.__version__,
            device="cpu",
            dtype="float64",
            threads=2,
        ),
        analysis=copy.deepcopy(ANALYSIS),
    )
    profile = None
    try:
        for assets in (64, 256, 1024):
            for quantiles in (False, True):
                head = "quantiles" if quantiles else "scalar"
                source = fixture(output / f"fixture-{assets}-{head}", assets, quantiles)
                records = {
                    arm: json.loads(path.read_text()) for arm, path in source["reports"].items()
                }
                assert len(records) == 6 and all(r["technical_fixture"] for r in records.values())
                inputs.append(
                    dict(
                        max_us_assets=assets,
                        cn_assets=[assets // 4, 3 * assets // 4],
                        head=head,
                        days_per_partition=48,
                        row_group_size=4096,
                        rows_per_control=records["baseline"]["predictions"]["evaluation"]["rows"],
                        reports={arm: sha256(path) for arm, path in source["reports"].items()},
                        study_sha256=sha256(source["study"]),
                    )
                )

                def fixtures_only(*_, current=source):
                    assert all(
                        path.is_relative_to(current["root"]) for path in current["reports"].values()
                    )
                    assert all(
                        json.loads(path.read_text()).get("technical_fixture")
                        == "manual_reports_without_model_execution"
                        for path in current["reports"].values()
                    )

                reference = None
                with pytest.MonkeyPatch.context() as scope:
                    scope.setattr(spaced_comparison, "require_learning_allowed", fixtures_only)
                    for repetition in range(4):
                        if time.perf_counter() - started >= 600:
                            result["status"] = "budget_exhausted"
                            atomic_json(output / "cost.json", result)
                            raise RuntimeError("Presupuesto técnico de 600 segundos agotado")
                        target = output / f"analysis-{assets}-{head}-{repetition}"
                        before = time.perf_counter()
                        report = spaced_comparison.analyze_spaced_runs(
                            source["reports"],
                            source["study"],
                            partition="evaluation",
                            output=target,
                        )
                        seconds = time.perf_counter() - before
                        measured = signature(report)
                        assert reference is None or measured == reference
                        reference = measured
                        samples.append(
                            dict(
                                max_us_assets=assets,
                                head=head,
                                repetition=repetition,
                                warmup=repetition == 0,
                                seconds=seconds,
                                prediction_rows=sum(
                                    r["rows"] for r in report["resources"].values()
                                ),
                                sessions_per_control=report["scores"]["baseline"]["sessions"],
                                comparison_sha256=sha256(target / "comparison.json"),
                                session_table_sha256=report["sessions"]["sha256"],
                            )
                        )
                        atomic_json(output / "cost.json", result)
                    if assets == 256 and quantiles:
                        assert time.perf_counter() - started < 600
                        profiler = cProfile.Profile()
                        before = time.perf_counter()
                        profiler.enable()
                        profiled = spaced_comparison.analyze_spaced_runs(
                            source["reports"],
                            source["study"],
                            partition="evaluation",
                            output=output / "profile-256-quantiles",
                        )
                        profiler.disable()
                        assert signature(profiled) == reference
                        profile = dict(
                            seconds=time.perf_counter() - before,
                            scope="Una ejecución adicional instrumentada, fuera de las medianas",
                            functions=profile_rows(profiler),
                        )
        result.update(
            status="completed",
            profile=profile,
            optimizer_attempts=optimizer_attempts,
            cuda_initialized=torch.cuda.is_initialized(),
            seconds_without_imports=time.perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            summaries=[
                dict(
                    max_us_assets=assets,
                    head=head,
                    repetitions=len(values),
                    median_seconds=statistics.median(values),
                    min_seconds=min(values),
                    max_seconds=max(values),
                )
                for assets in (64, 256, 1024)
                for head in ("scalar", "quantiles")
                for values in [
                    [
                        s["seconds"]
                        for s in samples
                        if s["max_us_assets"] == assets and s["head"] == head and not s["warmup"]
                    ]
                ]
            ],
            limits=[
                "Informes manuales replicados, sin cargar ni entrenar modelos.",
                "Seis controles y 48 días por partición. No son observaciones financieras reales.",
                "Tiempo por análisis: guardas de fixture, metadatos, hashes, lectura, "
                "puntuación, remuestreo y publicación. Excluye crear fixtures y "
                "comparar firmas después. El total sí incluye ambos.",
                "El proceso realiza 24 análisis ordinarios y uno perfilado. "
                "Las tres muestras medidas por caso no caracterizan p95/p99.",
                "CPU FP64. Sin GPU, transferencias ni energía medidas. "
                "El pico RSS pertenece al proceso completo, incluida preparación.",
            ],
        )
        atomic_json(output / "cost.json", result)
        print(json.dumps(dict(status=result["status"], summaries=result["summaries"])))
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
