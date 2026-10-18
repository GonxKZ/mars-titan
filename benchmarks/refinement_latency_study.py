"""Medir modelos ya seleccionados, sin modificar el estudio ni sus predicciones."""

import argparse
import json
import math
import platform
import signal
import time
from datetime import UTC, datetime
from pathlib import Path

from mars_titan.data.batches import read_bounded_table
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.training.learning_hold import require_learning_allowed

from . import refinement_latency
from .refinement_study import verify_prepared
from .refinement_study_policy import load_policy, selection_group
from .refinement_study_traces import completed_phase, reconstruct

CASES = ("k1", "k4cc", "policy")


def load_case(work, case):
    """Exigir ambos sellos y la evaluación completa de la primera semilla declarada."""
    if case not in CASES:
        raise ValueError("El caso no pertenece al protocolo de latencia")
    work = Path(work).resolve()
    prepared = verify_prepared(work)
    if prepared["configuration"]["cohort_size"] not in (64, 256):
        raise ValueError("La cohorte no pertenece al protocolo de latencia")
    if prepared.get("final_test_opened") is not False:
        raise ValueError("El test final debe permanecer cerrado")
    job = next(row for row in prepared["jobs"] if row["id"] == "linear_v1-42/evaluate-k1")
    group = selection_group(work, job)
    output, selected = work / "runs" / job["id"], work / "runs" / job["parent"]
    report = read_manifest(output / "run.json")[0]
    if (
        report.get("status") != "completed"
        or report.get("final_test_opened") is not False
        or report["job"] != job
        or report["selection_group"] != group
        or report["selection_sha256"] != sha256(selected / "phase.json")
    ):
        raise ValueError("La evaluación no conserva sus selecciones confirmadas")
    seal = read_manifest(selected / "phase.json")[0]
    policy = selected / seal["selected"]["id"] / "policy.json"
    if (
        not policy.resolve().is_relative_to(selected)
        or sha256(policy) != seal["selected"]["policy_sha256"]
    ):
        raise ValueError("La política seleccionada ha cambiado")
    folder = output if case == "policy" else output / "controls" / case
    phase = read_manifest(folder / "evaluation/phase.json")[0]
    expected = (
        report["phases"]["evaluation"]
        if case == "policy"
        else report["controls"][case]["evaluation"]
    )
    if phase != expected or completed_phase(folder / "evaluation", phase["identity"]) is None:
        raise ValueError("La fase medida no corresponde a la evaluación confirmada")
    return dict(
        prepared=prepared,
        job=job,
        phase=phase,
        policy_path=policy,
        prediction_path=folder / "evaluation/predictions.parquet",
        selection_group=group,
        selection_sha256=sha256(selected / "phase.json"),
        evaluation_sha256=sha256(output / "run.json"),
    )


class Deadline:
    def __init__(self, seconds):
        self.end, self.cancelled = time.monotonic() + seconds, False

    @property
    def requested(self):
        return self.cancelled or time.monotonic() >= self.end

    def request(self, *_args):
        self.cancelled = True


def _source(work, output):
    from mars_titan.data.input_policy import HISTORICAL_MASKED
    from mars_titan.memory.financial_session import FinancialPhase
    from mars_titan.training.corpus_inputs import CorpusDataset
    from mars_titan.training.titans_walk_forward import _sources

    dataset = CorpusDataset(work / "fold-018/view.json", input_policy=HISTORICAL_MASKED)
    phase = FinancialPhase(**read_manifest(work / "staged.json")[0]["phases"]["evaluation"])
    if phase.partition != "evaluation":
        raise ValueError("La medida requiere la partición de evaluación ya confirmada")
    sources = _sources(dataset, {"evaluation": phase}, output / "indices")
    return dataset, sources["evaluation"]


def _measure(work, output, case, request, stop, *, device="cuda:0"):
    from mars_titan.training.titans_walk_forward import checked_tables

    if device not in ("cpu", "cuda:0"):
        raise ValueError("La medida necesita un dispositivo explícito admitido")
    started = time.perf_counter()
    dataset, source = _source(work, output)
    expected = read_bounded_table(request["prediction_path"], max_rows=131072)
    expected_digest = refinement_latency.table_digest(expected)
    source_seconds = time.perf_counter() - started
    results = []
    for repeat in range(4):
        if stop.requested:
            raise TimeoutError("Se agotó el presupuesto de latencia antes de otra pasada")
        started = time.perf_counter()
        policy = load_policy(request["policy_path"], device=device) if case == "policy" else None
        reader_job = dict(request["job"], parent=request["job"]["reader"])
        engine, origin = reconstruct(
            work,
            reader_job,
            source,
            mode="policy" if policy else "fixed",
            policy=policy,
            route=None if policy else case,
            device=device,
        )
        if str(engine.device) != device:
            raise ValueError("La reconstrucción no conserva el dispositivo solicitado")
        if origin != request["phase"]["identity"]["origin"]:
            raise ValueError(
                "La reconstrucción cambió el núcleo, el lector o sus condiciones numéricas"
            )
        loading_seconds = time.perf_counter() - started
        record, rows = refinement_latency.measure_pass(
            engine, source, instrument=repeat > 0, stop=stop
        )
        checked_tables(rows, record["metrics"], dataset, source.phase.partition)
        if (
            refinement_latency.prediction_digest(rows) != expected_digest
            or record["metrics"] != request["phase"]["metrics"]
        ):
            raise ValueError(
                "La pasada no conserva exactamente las predicciones y contadores originales"
            )
        record.update(
            repeat=repeat,
            loading_seconds=loading_seconds,
            prediction_digest=expected_digest,
            exact_original_predictions=True,
            exact_original_metrics=True,
            origin=origin,
        )
        if repeat:
            record["summary"] = refinement_latency.summarize(record["events"], warmup_decisions=5)
        path = output / f"pass-{repeat}.json"
        atomic_json(path, record)
        results.append(
            dict(path=path.name, sha256=sha256(path), phase_seconds=record["phase_seconds"])
        )
        del engine, policy, rows
    baseline = results[0]["phase_seconds"]
    return dict(
        source_seconds=source_seconds,
        passes=results,
        instrumentation_phase_time_ratios=[r["phase_seconds"] / baseline for r in results[1:]],
        ratio_scope="time_ratio_includes_cache_and_order_effects_not_only_instrumentation",
        exact_original_predictions=True,
        exact_original_metrics=True,
    )


def run(work, output, case, *, max_seconds=1200):
    """Una invocación CUDA con destino nuevo. El presupuesto incluye carga y auditorías."""
    require_learning_allowed("la medición de desarrollo de los modelos seleccionados")
    if (
        type(max_seconds) not in (int, float)
        or not math.isfinite(max_seconds)
        or not 0 < max_seconds <= 1200
    ):
        raise ValueError("El presupuesto de latencia debe estar entre cero y 1200 segundos")
    work, output = Path(work).resolve(), Path(output)
    safe_destination(output)
    outside_source(work, output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("La medida de latencia necesita un destino nuevo")
    started = time.perf_counter()
    request = load_case(work, case)
    output.mkdir(parents=True)
    stop = Deadline(max_seconds - (time.perf_counter() - started))
    report = dict(
        status="running",
        case=case,
        prepared_sha256=sha256(work / "prepared.json"),
        selection_sha256=request["selection_sha256"],
        evaluation_sha256=request["evaluation_sha256"],
        max_seconds=max_seconds,
        device="cuda:0",
        optimizer_steps=0,
        final_test_opened=False,
        code={
            Path(__file__).name: sha256(Path(__file__)),
            "refinement_latency.py": sha256(Path(refinement_latency.__file__)),
        },
        timing_scope="function_including_admission_loading_indexing_replay_audits_and_output",
        started_at_utc=datetime.now(UTC).isoformat(),
    )
    atomic_json(output / "run.json", report)
    handlers = {sig: signal.signal(sig, stop.request) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        import torch

        from mars_titan.training.experiment_resources import GpuLease

        with GpuLease(max_ram=6 * 1024**3, max_vram=6 * 1024**3) as lease:
            torch.cuda.set_device(0)
            torch.set_num_threads(2)
            if torch.get_num_interop_threads() != 1:
                torch.set_num_interop_threads(1)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
            torch.backends.mha.set_fastpath_enabled(False)
            report.update(
                runtime=dict(
                    lease.record,
                    python=platform.python_version(),
                    cpu_threads=2,
                    precision="fp32_strict",
                )
            )
            report.update(_measure(work, output, case, request, stop))
            lease.check()
        report["status"] = "completed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        report.update(
            process_seconds=time.perf_counter() - started,
            finished_at_utc=datetime.now(UTC).isoformat(),
        )
        atomic_json(output / "run.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--max-seconds", type=float, default=1200)
    args = parser.parse_args()
    print(
        json.dumps(
            run(args.work, args.output, args.case, max_seconds=args.max_seconds), ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
