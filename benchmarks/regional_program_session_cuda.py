"""Medición privada de FinancialSession.step completo con fixtures manuales.

Sin --run imprime el protocolo y no importa Torch. La ejecución necesita una
salida nueva y un enlace nativo existente, indicado por --native o su variable.
El hijo técnico dispone de 100 segundos. El supervisor exterior corresponde
al llamante. No genera resultados científicos ni estima etiquetas residuales.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import statistics
import sys
import tempfile
import time
import traceback
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
HOLD = Path.home() / ".local/state/mars-titan/training-hold-2000.json"
HOLD_SHA256 = "8a658080c1ac4e63847eba58b177020339da6ec47021cf26d2701830073c87fd"
DEVICE = "cuda:0"
BATCHES = (1, 256)
VARIANTS = (False, True)
BLOCK_ROWS = 64
WARMUP, REPETITIONS, CHILD_SECONDS = 1, 3, 100
ALLOCATOR_CAP, HEADROOM = 512 * 1024**2, 256 * 1024**2


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def protocol():
    return dict(
        kind="financial_session_step_regional_program_fixture_cost_v1",
        script_sha256=sha256(Path(__file__).read_bytes()),
        batches=list(BATCHES),
        variants=["legacy_none", "fixed_summary_interleaved_v1"],
        dtype="torch.float64",
        device=DEVICE,
        hidden_size=32,
        head="quantile_head_v1",
        topology="ring",
        context="market_memory_v1_output_fixed_identity_times_0.2",
        collection=False,
        physical_block_rows=BLOCK_ROWS,
        blocks_per_decision={"1": 1, "256": 4},
        cases=4,
        warmup_sessions_per_case=WARMUP,
        measured_fresh_sessions_per_case=REPETITIONS,
        total_fresh_sessions=16,
        prefix_event=125,
        timed_event=126,
        fixture="test_financial_session_controls.flow_source(root, flows=US/T0000...)",
        labels="manual_fixture_no_estimation_literal_0.125_never_used_for_training",
        guards=[
            "contextmanager(test_frozen_financial.frozen_backend.__wrapped__)",
            "contextmanager(test_financial_session_controls.no_target_estimation.__wrapped__)",
            "install_optimizer_guard_with_pinned_global_hold",
        ],
        backend="load_native_existing_binary_ABI_and_Torch_version_checked_no_compilation",
        timed_scope="FinancialSession.step_for_event_126_plus_CUDA_completion_sync",
        includes=[
            "input_and_prefix_checks_inside_step",
            "whole_cohort_reservation_when_active",
            "physical_blocks_and_prediction",
            "context_commit",
            "native_commit",
            "confirmed_state_verification",
            "regional_reconciliation_and_compaction_when_active",
            "artifact_pruning",
        ],
        outside_timed_interval=[
            "manual_fixture_construction",
            "model_construction",
            "new_directory_and_session_creation",
            "warmup_event_125_and_prefix_snapshot",
            "prefix_session_close_and_reopen",
            "result_export_and_integrity_proofs",
            "final_session_close",
        ],
        outside_times_recorded=True,
        samples_summary="all_three_samples_min_median_max_no_p95_or_p99",
        exact_repeated_result="native_snapshot_fast_store_auxiliary_event_and_predictions_SHA256",
        pairing="exact_named_parameters_and_tensor_buffers_per_batch",
        integrity="model_inputs_and_CPU_CUDA_RNG_unchanged_reopen_preserves_prefix",
        deterministic_algorithms=True,
        tf32=False,
        autocast=False,
        mha_fastpath=False,
        cpu_threads=2,
        interop_threads=1,
        arrow_cpu_threads=2,
        arrow_io_threads=2,
        allocator_cap_bytes=ALLOCATOR_CAP,
        free_global_vram_headroom_bytes=HEADROOM,
        memory_scope="Torch_allocator_and_process_RSS_lifetime_peak_includes_setup_and_proofs",
        child_budget_seconds=CHILD_SECONDS,
        hold_sha256=HOLD_SHA256,
        optimizer_steps=0,
        target_estimators_called=0,
        real_corpus=False,
        scientific_evaluation=False,
        energy_measured=False,
        physical_IO_bytes_measured=False,
    )


def save(path, value):
    if not path.resolve().is_relative_to(BASE):
        raise ValueError("La salida debe permanecer junto al guion privado")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def check_hold():
    if HOLD.stat().st_size > 64 * 1024:
        raise ValueError("La guarda de aprendizaje supera el límite de lectura")
    raw = HOLD.read_bytes()
    if sha256(raw) != HOLD_SHA256 or json.loads(raw).get("training_allowed") is not False:
        raise ValueError("La guarda fijada no conserva cerrado el aprendizaje")
    return sha256(raw)


def load_helpers(root):
    path = root / "benchmarks/regional_decision_cuda.py"
    spec = importlib.util.spec_from_file_location("_regional_session_cost_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError("No se puede cargar el helper de medición")
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    if (helper.DEVICE, helper.ALLOCATOR_CAP, helper.HEADROOM) != (DEVICE, ALLOCATOR_CAP, HEADROOM):
        raise ValueError("El helper no conserva dispositivo y límites de memoria")
    return helper


def manifest(root, helper):
    result = helper.manifest(root)
    result["benchmarks/regional_decision_cuda.py"] = sha256(
        (root / "benchmarks/regional_decision_cuda.py").read_bytes()
    )
    return result


def input_proof(source):
    digest, logical = hashlib.sha256(), 0
    for event in (125, 126):
        for batch in source["batches"][event]:
            batch.verify()
            digest.update(batch.input_digest.encode())
            for name in sorted(batch.inputs):
                value = batch.inputs[name]
                digest.update(name.encode())
                digest.update(value.tobytes())
                logical += value.nbytes
            digest.update(batch.presence.tobytes())
            logical += batch.presence.nbytes
    return digest.hexdigest(), logical


def prediction_rows(result):
    return [
        dict(
            id=row.id,
            asset=row.asset,
            task=row.task.name,
            horizon=row.task.horizon,
            generation=row.generation,
            decision_at=row.decision_at,
            value=row.value,
        )
        for row in result.predictions
    ]


def one_session(
    torch,
    helper,
    native,
    session,
    data,
    source,
    out,
    *,
    batch_size,
    active,
    index,
    warmup,
    deadline,
):
    helper.check_deadline(deadline)
    name = "active" if active else "legacy"
    directory = (
        out / "sessions" / f"b{batch_size}-{name}-{'warmup' if warmup else 'sample'}-{index}"
    )
    options = dict(block_rows=BLOCK_ROWS, collection=False)
    if active:
        from test_regional_program_session import limits

        options["regional_work_limits"] = limits()
    record = dict(directory=str(directory), warmup=warmup, status="incomplete")
    cpu_rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    run = None
    try:
        started = time.perf_counter()
        run = session(native, directory, data, **options)
        torch.cuda.synchronize(DEVICE)
        record["creation_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        run.step(source["batches"][125], [], kind="warmup")
        torch.cuda.synchronize(DEVICE)
        record["prefix_event_125_seconds"] = time.perf_counter() - started
        before = run.snapshot()
        record["prefix_snapshot_sha256"] = helper.proof(torch, before)[0]
        started = time.perf_counter()
        run.close()
        run = None
        record["prefix_close_seconds"] = time.perf_counter() - started
        helper.check_deadline(deadline)
        started = time.perf_counter()
        run = session(native, directory, data, resume=True, **options)
        torch.cuda.synchronize(DEVICE)
        record["reopen_prefix_seconds"] = time.perf_counter() - started
        if run.snapshot() != before:
            raise AssertionError("La reapertura cambió el prefijo confirmado")
        check_hold()
        helper.check_memory(torch)
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        allocated_before = torch.cuda.memory_allocated(DEVICE)
        reserved_before = torch.cuda.memory_reserved(DEVICE)
        record["ram_before_step"] = helper.ram()
        started = time.perf_counter()
        result = run.step(source["batches"][126], [])
        torch.cuda.synchronize(DEVICE)
        record["step_seconds"] = time.perf_counter() - started
        memory, rss = helper.check_memory(torch), helper.ram()
        record.update(
            **memory,
            **rss,
            peak_allocated_extra_bytes=memory["peak_allocated_bytes"] - allocated_before,
            peak_reserved_extra_bytes=memory["peak_reserved_bytes"] - reserved_before,
        )
        predictions = prediction_rows(result)
        snapshot, fast, auxiliary = run.snapshot(), run.fast_state(), run.mechanism_state()
        event = run.mechanism_event(snapshot["cursor"])
        if len(predictions) != batch_size or snapshot["observed"] != 2 * batch_size:
            raise AssertionError("La sesión no conserva las observaciones y emisiones previstas")
        if sum(row["observed_steps"] for row in fast["references"].values()) != 2 * batch_size:
            raise AssertionError("El estado rápido no conserva dos avances por flujo")
        result_sha, logical, _ = helper.proof(
            torch,
            dict(
                snapshot=snapshot,
                fast_store=fast,
                auxiliary=auxiliary,
                event=event,
                predictions=predictions,
            ),
        )
        record.update(
            result_sha256=result_sha,
            result_logical_tensor_bytes=logical,
            snapshot_sha256=helper.proof(torch, snapshot)[0],
            observed_work=event["payload"]["work"],
            reopen_preserved_prefix=True,
        )
        if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(
            cuda_rng, torch.cuda.get_rng_state(0)
        ):
            raise AssertionError("La sesión consumió el RNG compartido de Torch")
        record.update(status="passed", torch_rng_unchanged=True)
        helper.check_deadline(deadline)
        return record
    finally:
        if run is not None:
            started = time.perf_counter()
            run.close()
            record["final_close_seconds"] = time.perf_counter() - started


def measure(torch, helper, native, root, out, report, deadline):
    import pyarrow as pa
    from test_financial_session_controls import flow_source, no_target_estimation
    from test_frozen_financial import frozen_backend
    from test_regional_program_session import engines, session

    from mars_titan.training.learning_hold import install_optimizer_guard

    pa.set_cpu_count(2)
    pa.set_io_thread_count(2)

    def forbidden_optimizer(reason):
        raise RuntimeError(reason)

    guard = install_optimizer_guard(forbidden_optimizer, path=HOLD)
    if guard is None:
        raise RuntimeError("No se pudo instalar la protección de optimizadores")
    try:
        with (
            contextmanager(frozen_backend.__wrapped__)(),
            contextmanager(no_target_estimation.__wrapped__)(),
        ):
            report["fixture_guards_active"] = True
            for batch_size in BATCHES:
                helper.check_deadline(deadline)
                flows = tuple(f"US/T{i:04d}" for i in range(batch_size))
                started = time.perf_counter()
                source = flow_source(out / f"fixture-b{batch_size}", flows=flows)
                source_seconds = time.perf_counter() - started
                if source["label_origin"] != "manual_fixture_no_estimation":
                    raise AssertionError("La fuente no utiliza las etiquetas manuales previstas")
                inputs_sha, input_bytes = input_proof(source)
                models, common = {}, None
                for active in VARIANTS:
                    started = time.perf_counter()
                    data = engines(source, DEVICE, active=active, online=True, calibrated=False)
                    torch.cuda.synchronize(DEVICE)
                    creation = time.perf_counter() - started
                    model = data[3].predictor
                    tensors = dict(model.named_parameters()) | {
                        f"buffer/{name}": value for name, value in model.named_buffers()
                    }
                    named_sha, _, retained = helper.proof(torch, tensors, keep=True)
                    if common is None:
                        common = retained
                    else:
                        helper.strict_equal(torch, common, retained)
                    models[active] = (
                        data,
                        creation,
                        named_sha,
                        helper.proof(torch, model.state_dict())[0],
                    )
                    helper.check_memory(torch)
                for active in VARIANTS:
                    data, creation, named_sha, model_sha = models[active]
                    entry = dict(
                        batch=batch_size,
                        active=active,
                        dtype="torch.float64",
                        physical_block_rows=BLOCK_ROWS,
                        physical_blocks=1 if batch_size == 1 else 4,
                        shared_source_creation_seconds=source_seconds,
                        model_creation_seconds=creation,
                        input_sha256=inputs_sha,
                        input_logical_bytes=input_bytes,
                        paired_named_tensors_sha256=named_sha,
                        model_state_sha256=model_sha,
                        consumer_identity=data[3].identity(),
                        status="incomplete",
                        warmup=None,
                        samples=[],
                    )
                    report["cases"].append(entry)
                    reference = None
                    for index in range(WARMUP + REPETITIONS):
                        warmup = index < WARMUP
                        record = one_session(
                            torch,
                            helper,
                            native,
                            session,
                            data,
                            source,
                            out,
                            batch_size=batch_size,
                            active=active,
                            index=index,
                            warmup=warmup,
                            deadline=deadline,
                        )
                        if reference is None:
                            reference = record["result_sha256"]
                        elif record["result_sha256"] != reference:
                            raise AssertionError("Una sesión nueva cambió el resultado completo")
                        if warmup:
                            entry["warmup"] = record
                        else:
                            entry["samples"].append(record)
                        if (
                            input_proof(source)[0] != inputs_sha
                            or helper.proof(torch, data[3].predictor.state_dict())[0] != model_sha
                        ):
                            raise AssertionError("La sesión cambió las entradas o los parámetros")
                        data[3].verify(strong=True)
                        save(out / "measurement.json", report)
                    seconds = [row["step_seconds"] for row in entry["samples"]]
                    entry.update(
                        status="passed",
                        min_ms=1000 * min(seconds),
                        median_ms=1000 * statistics.median(seconds),
                        max_ms=1000 * max(seconds),
                        mean_rows_per_second=batch_size / statistics.mean(seconds),
                        exact_repeated_results=True,
                        parameters_inputs_unchanged=True,
                    )
                    save(out / "measurement.json", report)
                    print(f"B={batch_size} active={active}: 3 medidas completas", flush=True)
                del models, source, common, data, tensors, retained
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        guard.remove()


def run(root, out, native_path):
    started = time.monotonic()
    deadline = started + CHILD_SECONDS
    if out.exists() or not out.is_relative_to(BASE) or out.is_relative_to(root):
        raise ValueError("Se necesita una salida privada nueva fuera del repositorio")
    out.mkdir(parents=True, exist_ok=False)
    helper = load_helpers(root)
    report = dict(protocol=protocol(), status="incomplete", started_at_utc=now(), cases=[])

    def timed_out(signum, frame):
        raise TimeoutError("La medición durable agotó el límite de 100 segundos")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline - time.monotonic()))
    try:
        report["hold_before"] = check_hold()
        save(out / "protocol.json", protocol())
        pinned = manifest(root, helper)
        save(out / "source-manifest.json", pinned)
        report["hardware_before"] = helper.nvidia()
        os.environ.update(
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MARS_TITAN_REQUIRE_CUDA="1",
            MARS_TITAN_EPISODIC_NATIVE=str(native_path),
            PYTHONDONTWRITEBYTECODE="1",
        )
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. No se permite sustitución por CPU")
        torch.cuda.set_device(DEVICE)
        if "4070" not in torch.cuda.get_device_name(0):
            raise RuntimeError("cuda:0 no es la RTX 4070 prevista")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.mha.set_fastpath_enabled(False)
        if torch.is_autocast_enabled("cuda") or torch.is_inference_mode_enabled():
            raise ValueError("La medición exige FP64 fuera de autocast e inference_mode")
        free, total = torch.cuda.mem_get_info(DEVICE)
        if free < ALLOCATOR_CAP + HEADROOM:
            raise MemoryError("No hay 512 MiB para el asignador y 256 MiB libres de reserva")
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP / total, DEVICE)
        for part in ("src", ".", "tests/models/titans", "tests/memory"):
            sys.path.insert(0, str(root / part))
        from mars_titan.memory.native_backend import load_native

        native = load_native(native_path)
        report["environment"] = dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            torch=str(torch.__version__),
            torch_git=torch.version.git_version,
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            numpy=importlib.metadata.version("numpy"),
            pyarrow=importlib.metadata.version("pyarrow"),
            pytest=importlib.metadata.version("pytest"),
            gpu=torch.cuda.get_device_name(0),
            gpu_properties=str(torch.cuda.get_device_properties(0)),
            native_path=str(native_path),
            native_binary_sha256=native.binary_sha256,
            native_abi=native.abi_version,
            native_torch_version=native.torch_version,
            cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            allocator_fraction=torch.cuda.get_per_process_memory_fraction(DEVICE),
            free_vram_before_bytes=free,
            total_vram_bytes=total,
            tf32=False,
            autocast=False,
            mha_fastpath=False,
        )
        save(out / "measurement.json", report)
        measure(torch, helper, native, root, out, report, deadline)
        if len(report["cases"]) != 4 or any(len(case["samples"]) != 3 for case in report["cases"]):
            raise AssertionError("Faltan las doce muestras previstas")
        if manifest(root, helper) != pinned:
            raise ValueError("Las fuentes cambiaron durante la medición")
        with native_path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != native.binary_sha256:
                raise ValueError("El enlace nativo cambió durante la medición")
        report.update(
            status="passed",
            source_manifest_unchanged=True,
            hold_after=check_hold(),
            hardware_after=helper.nvidia(),
        )
        helper.check_deadline(deadline)
    except BaseException as error:
        report.update(
            status="incomplete" if isinstance(error, TimeoutError) else "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        try:
            report["hold_after"] = check_hold()
        except (OSError, ValueError) as error:
            report.update(hold_after=None, hold_error=str(error), status="failed")
        report.update(
            hold_unchanged=report.get("hold_before") == report.get("hold_after") == HOLD_SHA256,
            finished_at_utc=now(),
            total_process_seconds=time.monotonic() - started,
            ram_final=helper.ram(),
        )
        if report["total_process_seconds"] > CHILD_SECONDS:
            report.update(status="incomplete", budget_exceeded=True)
        save(out / "measurement.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--native", type=Path)
    options = parser.parse_args()
    if not options.run:
        print(json.dumps(protocol(), ensure_ascii=False, indent=2))
        return
    if options.output is None:
        parser.error("--run necesita una salida privada nueva")
    selected = options.native or os.environ.get("MARS_TITAN_EPISODIC_NATIVE")
    if not selected or not Path(selected).is_file():
        parser.error(
            "Se necesita un enlace nativo existente mediante --native o MARS_TITAN_EPISODIC_NATIVE"
        )
    run(
        options.repository.resolve(strict=True),
        options.output.resolve(),
        Path(selected).resolve(strict=True),
    )


if __name__ == "__main__":
    main()
