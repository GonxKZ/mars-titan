"""Medir prepare completo con programas regionales y un prefijo técnico confirmado.

Sin --run solo imprime el protocolo y no importa Torch. Para medir hay que copiar
el guion fuera del repositorio y usar --run --repository y una salida privada nueva.
El supervisor limita su proceso hijo a 180 segundos y el hijo dispone de 100.

Las cuatro condiciones comparten exactamente parámetros, entradas y formas. Cada
identidad prepara sus dos emisiones previas y conserva su propio estado resultante.
El intervalo incluye consulta de contexto, codificadores, fusión, MAC, cabeza y
propuesta de estado. La comprobación por hashes queda fuera del intervalo medido.
No mide colección, sesión nativa, persistencia, feedback ni calidad predictiva.
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
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import fields
from pathlib import Path

BASE = Path(__file__).resolve().parent
DEVICE = "cuda:0"
DTYPES = ("float32", "float64")
BATCHES = (1, 256)
VARIANTS = ("none", "summary_at_end_v1", "summary_interleaved_v1", "context_norm_v1")
WARMUP, REPETITIONS = 3, 10
CHILD_SECONDS, SUPERVISOR_SECONDS = 100, 180
ALLOCATOR_CAP, HEADROOM = 512 * 1024**2, 256 * 1024**2
HOLD = Path.home() / ".local/state/mars-titan/training-hold-2000.json"
HOLD_SHA256 = "8a658080c1ac4e63847eba58b177020339da6ec47021cf26d2701830073c87fd"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def protocol():
    """Fijar condiciones, exclusiones y comprobaciones antes de medir."""
    return dict(
        kind="financial_prepare_regional_program_fixture_cost_v1",
        script_sha256=sha256(Path(__file__).read_bytes()),
        device=DEVICE,
        dtypes=list(DTYPES),
        batches=list(BATCHES),
        hidden_size=32,
        topology="ring",
        operator_default=dict(cycles=2, local_steps=2, summary=True),
        variants=list(VARIANTS),
        orders=[list(VARIANTS), list(reversed(VARIANTS))],
        condition_records=32,
        warmup_per_record=WARMUP,
        repetitions_per_record=REPETITIONS,
        timed_prepare_calls=32 * (WARMUP + REPETITIONS),
        prefix_prepare_calls=32,
        predictor_fixture="test_market_memory.pt4(MARKET_MEMORY), seed=42, scalar_head",
        input_fixture="test_context_selection.batch_on, events=0,1,2, seed=23",
        fp32_inputs="rebuild_DecisionBatch_from_the_same_original_float32_fixture_values",
        context=dict(
            output="fixed_identity_times_0.2_loaded_before_measurement",
            prefix="two_prepare_and_commit_context_emissions_per_identity",
            frozen="own_state_and_confirmed_context_after_event_1_reused_for_event_2",
            common_values_across_variants=False,
            threshold=1e-9,
        ),
        paired_parameters="all_named_parameters_and_tensor_buffers_exactly_equal_per_dtype",
        timed_scope="FinancialPredictor.prepare_plus_completion_sync",
        includes=[
            "prepare_input_and_state_checks",
            "context_query",
            "fixture_encoders",
            "regional_fusion",
            "MAC",
            "head",
            "next_state",
        ],
        excludes=[
            "initial_batch_construction_validation_and_transfer",
            "prefix_construction",
            "context_commit",
            "proof_export",
            "collector",
            "native_session",
            "persistence",
            "feedback",
        ],
        native_session="not_measured_separate_durable_replay_and_fixture_preparation_required",
        reader_K=None,
        reader_J=None,
        program_cycles="independent_of_reader_refinements_and_recurrence",
        deterministic_algorithms=True,
        tf32=False,
        autocast=False,
        precision_relaxed=False,
        result_check="full_result_exact_SHA256_each_call_and_strict_equal_first_two_per_record",
        immutability="parameters_buffers_inputs_fast_state_context_and_CPU_CUDA_RNG",
        cross_variant_numerical_parity_required=False,
        percentiles="linear_interpolation_descriptive_ten_samples_not_tail_estimates",
        throughput="batch_rows_divided_by_mean_synchronized_prepare_seconds",
        allocator_cap_bytes=ALLOCATOR_CAP,
        free_global_vram_headroom_bytes=HEADROOM,
        memory_scope="Torch_allocator_and_process_RSS_including_other_resident_cases_and_proofs",
        headroom_checks="before_and_after_each_prepare_not_continuous",
        child_budget_seconds=CHILD_SECONDS,
        supervisor_budget_seconds=SUPERVISOR_SECONDS,
        hold_sha256=HOLD_SHA256,
        optimizer_steps=0,
        shared_parameter_updates=0,
        real_corpus=False,
        scientific_evaluation=False,
        energy_measured=False,
        monetary_cost_measured=False,
        physical_transfers_measured=False,
    )


def helpers_on(root):
    """Cargar solo utilidades que no dependen del BASE de su guion original."""
    path = root / "benchmarks/regional_decision_cuda.py"
    spec = importlib.util.spec_from_file_location("_regional_program_measurement_helpers", path)
    if spec is None or spec.loader is None:
        raise ImportError("No se puede cargar el helper de medición regional")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if (module.DEVICE, module.ALLOCATOR_CAP, module.HEADROOM) != (DEVICE, ALLOCATOR_CAP, HEADROOM):
        raise ValueError("El helper no conserva dispositivo y límites de memoria")
    return module


def save(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(BASE):
        raise ValueError("La salida debe quedar junto a la copia privada del guion")
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
    value = json.loads(raw)
    if sha256(raw) != HOLD_SHA256 or value.get("training_allowed") is not False:
        raise ValueError("La guarda fijada no conserva cerrado el aprendizaje")
    return sha256(raw)


def manifest(root, helpers):
    result = helpers.manifest(root)
    for name in ("benchmarks/regional_decision_cuda.py", "benchmarks/regional_program_cuda.py"):
        result[name] = sha256((root / name).read_bytes())
    if result["benchmarks/regional_program_cuda.py"] != sha256(Path(__file__).read_bytes()):
        raise ValueError("La copia privada no coincide con el benchmark del repositorio")
    return result


def batch_for(torch, model, flows, event):
    import numpy as np
    from test_context_selection import batch_on

    from mars_titan.models.titans.financial_inputs import DecisionBatch

    batch = batch_on(model, flows, event)
    if model.head.weight.dtype == torch.float64:
        return batch
    raw = dict(
        inputs={
            name: value.cpu().numpy().astype(np.float32) for name, value in batch.inputs.items()
        },
        presence=batch.presence.cpu().numpy(),
        sample_ids=list(batch.sample_ids),
        prediction_at=np.array(batch.prediction_at, dtype="datetime64[us]"),
        input_available_at=np.array(batch.input_available_at, dtype="datetime64[us]"),
    )
    rebuilt = DecisionBatch.from_corpus(
        raw, model.config.inputs, device=DEVICE, dtype=torch.float32
    )
    if rebuilt.input_digest != batch.input_digest:
        raise AssertionError("La conversión FP32 cambió los inputs originales de la fixture")
    return rebuilt


def models_for(torch, helpers, dtype):
    from test_market_memory import MARKET_MEMORY, pt4

    from mars_titan.models.titans.regional_execution import RegionalExecutionConfig
    from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

    models, paired, shared_sha = {}, None, None
    for variant in VARIANTS:
        if variant == "none":
            program = None
        elif variant == "context_norm_v1":
            program = RegionalExecutionConfig(selection="context_norm_v1", context_threshold=1e-9)
        else:
            program = RegionalExecutionConfig(program=variant)
        model = pt4(
            MARKET_MEMORY,
            regional_fusion=RegionalFusionConfig(
                topology="ring", cycles=2, local_steps=2, summary=True
            ),
            regional_execution=program,
        ).to(device=DEVICE, dtype=dtype)
        weights = model.state_dict()
        weights["session_context.output"] = torch.eye(32, device=DEVICE, dtype=dtype) * 0.2
        model.load_state_dict(weights)
        model.eval().requires_grad_(False)
        tensors = dict(model.named_parameters()) | {
            f"buffer/{name}": value for name, value in model.named_buffers()
        }
        tensor_sha, _, retained = helpers.proof(torch, tensors, keep=True)
        if paired is None:
            paired, shared_sha = retained, tensor_sha
        else:
            helpers.strict_equal(torch, paired, retained)
        models[variant] = model
        helpers.check_memory(torch)
    if len({model._config_id() for model in models.values()}) != len(VARIANTS):
        raise AssertionError("Las cuatro condiciones no conservan identidades distintas")
    return models, shared_sha


def measure_case(torch, helpers, out, report, models, dtype_name, batch_size, deadline):
    helpers.check_deadline(deadline)
    flows = tuple(f"US/F{i:04d}" for i in range(batch_size))
    batches = [batch_for(torch, models["none"], flows, event) for event in range(3)]
    payload = [
        {
            field.name: getattr(batch, field.name)
            for field in fields(batch)
            if field.name != "_versions"
        }
        for batch in batches
    ]
    batch_sha, batch_bytes, _ = helpers.proof(torch, payload)
    entry = dict(
        dtype=dtype_name,
        batch=batch_size,
        batch_sha256=batch_sha,
        batch_tensor_bytes=batch_bytes,
        status="incomplete",
        models={},
        records=[],
    )
    report["cases"].append(entry)
    frozen, references = {}, {}
    for variant, model in models.items():
        helpers.check_deadline(deadline)
        state, context = model.initial_state(flows), model.initial_context()
        for batch in batches[:2]:
            helpers.check_memory(torch)
            prepared = model.prepare(
                batch, state, context=context, context_selection=(True,) * batch_size
            )
            state = prepared.next_state
            context = model.commit_context(prepared.next_context)
            del prepared
            helpers.check_deadline(deadline)
        if any(value is not None for value in context.pending_tokens) or not any(
            at >= 0 for at in context.committed_at
        ):
            raise AssertionError("El prefijo no está confirmado y libre de escrituras pendientes")
        if not torch.equal(
            state.observed_steps.cpu(), torch.full((batch_size,), 2, dtype=torch.int64)
        ):
            raise AssertionError("El prefijo no contiene exactamente dos emisiones por flujo")
        state_sha, state_bytes, _ = helpers.proof(torch, dict(fast=state, context=context))
        parameter_sha, parameter_bytes, _ = helpers.proof(torch, model.state_dict())
        frozen[variant] = (state, context, state_sha, parameter_sha)
        entry["models"][variant] = dict(
            config_id=model._config_id(),
            parameter_id=model._parameter_id,
            extra_state=model.get_extra_state(),
            initial_state_sha256=state_sha,
            initial_state_tensor_bytes=state_bytes,
            full_parameter_state_sha256=parameter_sha,
            parameter_tensor_bytes=parameter_bytes,
        )
    rng_cpu, rng_cuda = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone()
    entry["rng_before_sha256"] = dict(
        cpu=helpers.proof(torch, rng_cpu)[0], cuda_0=helpers.proof(torch, rng_cuda)[0]
    )
    batch = batches[2]
    for pass_index in range(2):
        order = list(VARIANTS if pass_index == 0 else reversed(VARIANTS))
        for position, variant in enumerate(order):
            model = models[variant]
            state, context, _, _ = frozen[variant]
            record = dict(
                variant=variant,
                pass_index=pass_index,
                position=position,
                actual_order=order,
                status="incomplete",
                warmup_seconds=[],
                samples=[],
                ram_before=helpers.ram(),
            )
            entry["records"].append(record)
            first_cpu = None
            for iteration in range(WARMUP + REPETITIONS):
                helpers.check_deadline(deadline)
                check_hold()
                helpers.check_memory(torch)
                torch.cuda.synchronize(DEVICE)
                torch.cuda.reset_peak_memory_stats(DEVICE)
                before_a = torch.cuda.memory_allocated(DEVICE)
                before_r = torch.cuda.memory_reserved(DEVICE)
                started = time.perf_counter()
                result = model.prepare(
                    batch, state, context=context, context_selection=(True,) * batch_size
                )
                torch.cuda.synchronize(DEVICE)
                elapsed = time.perf_counter() - started
                memory = helpers.check_memory(torch)
                rss = helpers.ram()
                result_sha, result_bytes, cpu = helpers.proof(torch, result, keep=iteration < 2)
                if references.setdefault(variant, result_sha) != result_sha:
                    raise AssertionError("Una repetición u orden cambió el resultado completo")
                if iteration == 0:
                    first_cpu = cpu
                elif iteration == 1:
                    helpers.strict_equal(torch, first_cpu, cpu)
                    first_cpu = None
                if not torch.equal(
                    result.next_state.observed_steps.cpu(),
                    torch.full((batch_size,), 3, dtype=torch.int64),
                ):
                    raise AssertionError("La medida no añadió exactamente una emisión por flujo")
                if model.regional_execution is not None:
                    rows = model.regional_execution.export_trace(
                        result.regional_trace, model.regional_fusion
                    )
                    counts = model.regional_execution.verify_exported_traces(
                        rows, model.regional_fusion
                    )
                    if not all(row["context_norm"] > 0 for row in rows):
                        raise AssertionError("La consulta no usa el contexto confirmado no vacío")
                    record["observed_regional_work"] = counts
                    record["observed_programs"] = sorted({row["program"] for row in rows})
                    record["context_norm_range"] = [
                        min(row["context_norm"] for row in rows),
                        max(row["context_norm"] for row in rows),
                    ]
                record.update(result_sha256=result_sha, logical_result_tensor_bytes=result_bytes)
                if iteration < WARMUP:
                    record["warmup_seconds"].append(elapsed)
                else:
                    record["samples"].append(
                        dict(
                            seconds=elapsed,
                            **memory,
                            **rss,
                            peak_allocated_extra_bytes=memory["peak_allocated_bytes"] - before_a,
                            peak_reserved_extra_bytes=memory["peak_reserved_bytes"] - before_r,
                        )
                    )
                del result, cpu
                helpers.check_deadline(deadline)
            seconds = [sample["seconds"] for sample in record["samples"]]
            record.update(
                status="passed",
                p50_ms=1000 * helpers.percentile(seconds, 0.5),
                p95_ms=1000 * helpers.percentile(seconds, 0.95),
                p99_ms=1000 * helpers.percentile(seconds, 0.99),
                mean_ms=1000 * statistics.mean(seconds),
                sample_stdev_ms=1000 * statistics.stdev(seconds),
                rows_per_second=batch_size / statistics.mean(seconds),
                ram_after=helpers.ram(),
            )
            save(out / "measurement.json", report)
            print(
                f"dtype={dtype_name} B={batch_size} order={pass_index} variant={variant}: passed",
                flush=True,
            )
    for variant, model in models.items():
        state, context, state_sha, parameter_sha = frozen[variant]
        if (
            helpers.proof(torch, dict(fast=state, context=context))[0] != state_sha
            or helpers.proof(torch, model.state_dict())[0] != parameter_sha
        ):
            raise AssertionError("La medida modificó el estado inicial o los parámetros")
        model.verify_parameter_identity()
    if (
        helpers.proof(torch, payload)[0] != batch_sha
        or not torch.equal(rng_cpu, torch.random.get_rng_state())
        or not torch.equal(rng_cuda, torch.cuda.get_rng_state(0))
    ):
        raise AssertionError("La medida modificó las entradas o el RNG")
    entry.update(
        status="passed",
        exact_repeated_results=True,
        parameters_inputs_state_context_rng_unchanged=True,
    )
    save(out / "measurement.json", report)


def worker(root, out, outer_deadline):
    helpers = helpers_on(root)
    started = time.monotonic()
    deadline = min(outer_deadline, started + CHILD_SECONDS)
    report = dict(protocol=protocol(), status="incomplete", started_at_utc=helpers.now(), cases=[])

    def timed_out(signum, frame):
        raise TimeoutError("El proceso de medida agotó el límite interno de 100 segundos")

    old_alarm = signal.signal(signal.SIGALRM, timed_out)
    signal.setitimer(signal.ITIMER_REAL, max(0.01, deadline - time.monotonic()))
    try:
        pinned = json.loads((out / "source-manifest.json").read_text())
        if (
            manifest(root, helpers) != pinned
            or json.loads((out / "protocol.json").read_text()) != protocol()
        ):
            raise ValueError("Cambió el protocolo o las fuentes antes de importar Torch")
        check_hold()
        report["hardware_before"] = helpers.nvidia()
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. No se permite sustituirla por CPU")
        torch.cuda.set_device(DEVICE)
        if "4070" not in torch.cuda.get_device_name(0):
            raise RuntimeError("cuda:0 no es la RTX 4070 prevista")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.is_autocast_enabled("cuda") or torch.is_inference_mode_enabled():
            raise ValueError("La medida exige precisión explícita fuera de inference_mode")
        free, total = torch.cuda.mem_get_info(DEVICE)
        if free < ALLOCATOR_CAP + HEADROOM:
            raise MemoryError("No hay 512 MiB para el asignador y 256 MiB de reserva libre")
        torch.cuda.set_per_process_memory_fraction(ALLOCATOR_CAP / total, DEVICE)
        report["environment"] = dict(
            python=sys.version,
            executable=sys.executable,
            platform=platform.platform(),
            torch=str(torch.__version__),
            torch_git=torch.version.git_version,
            numpy=importlib.metadata.version("numpy"),
            cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(),
            gpu=torch.cuda.get_device_name(0),
            gpu_properties=str(torch.cuda.get_device_properties(0)),
            visible_gpu_count=torch.cuda.device_count(),
            used_devices=[DEVICE],
            free_vram_before_bytes=free,
            total_vram_bytes=total,
            allocator_fraction=torch.cuda.get_per_process_memory_fraction(DEVICE),
            cuda_memory_api_source_sha256=sha256(Path(torch.cuda.memory.__file__).read_bytes()),
            cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            deterministic_algorithms=True,
            float32_matmul_precision=torch.get_float32_matmul_precision(),
            tf32=False,
            autocast=False,
            mha_fastpath=False,
        )
        for part in ("src", ".", "tests/models/titans", "tests/memory"):
            sys.path.insert(0, str(root / part))
        for dtype_name in DTYPES:
            helpers.check_deadline(deadline)
            models, shared_sha = models_for(torch, helpers, getattr(torch, dtype_name))
            report.setdefault("paired_named_tensor_sha256", {})[dtype_name] = shared_sha
            for batch_size in BATCHES:
                measure_case(torch, helpers, out, report, models, dtype_name, batch_size, deadline)
                gc.collect()
                torch.cuda.empty_cache()
            del models
            gc.collect()
            torch.cuda.empty_cache()
        count = sum(len(case["records"]) for case in report["cases"])
        if count != 32 or manifest(root, helpers) != pinned:
            raise AssertionError("Faltan condiciones o cambiaron las fuentes durante la medida")
        report.update(
            status="passed",
            actual_condition_records=count,
            source_manifest_unchanged=True,
            hold_sha256_final=check_hold(),
            hardware_after=helpers.nvidia(),
        )
        helpers.check_deadline(deadline)
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
        signal.signal(signal.SIGALRM, old_alarm)
        report.update(
            finished_at_utc=helpers.now(),
            child_process_seconds=time.monotonic() - started,
            ram_final=helpers.ram(),
        )
        if report["child_process_seconds"] > CHILD_SECONDS:
            report.update(status="incomplete", child_budget_exceeded=True)
        save(out / "measurement.json", report)


def supervise(root, out):
    helpers = helpers_on(root)
    started = time.monotonic()
    deadline = started + SUPERVISOR_SECONDS
    if (
        Path(__file__).resolve().is_relative_to(root)
        or not out.is_relative_to(BASE)
        or out.is_relative_to(root)
        or out.exists()
    ):
        raise ValueError("Se necesita copia privada y una salida externa nueva junto al guion")
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="incomplete",
        started_at_utc=helpers.now(),
        protocol=protocol(),
        repository=str(root),
        output=str(out),
        child_started=False,
        hold_before=None,
    )
    process = None
    try:
        receipt["hold_before"] = check_hold()
        save(out / "protocol.json", protocol())
        save(out / "source-manifest.json", manifest(root, helpers))
        receipt["hardware_before"] = helpers.nvidia()
        environment = dict(os.environ)
        environment.update(
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MARS_TITAN_REQUIRE_CUDA="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            str(Path(__file__).resolve()),
            "--run",
            "--worker",
            "--repository",
            str(root),
            "--output",
            str(out),
            "--deadline",
            repr(deadline - 2),
        ]
        receipt["command"] = command
        save(out / "execution.json", receipt)
        with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
            process = subprocess.Popen(
                command,
                cwd=root,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
            )
            receipt.update(child_started=True, child_pid=process.pid)
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic() - 2))
            except subprocess.TimeoutExpired as error:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=0.5)
                raise TimeoutError("El supervisor agotó el límite de 180 segundos") from error
        receipt["returncode"] = process.returncode
        measurement = json.loads((out / "measurement.json").read_text())
        if (
            process.returncode
            or measurement["status"] != "passed"
            or measurement.get("actual_condition_records") != 32
        ):
            raise RuntimeError("El hijo no confirmó los 32 registros completos")
        if manifest(root, helpers) != json.loads((out / "source-manifest.json").read_text()):
            raise ValueError("Las fuentes cambiaron durante la ejecución")
        receipt.update(
            status="passed",
            source_manifest_unchanged=True,
            measurement_sha256=sha256((out / "measurement.json").read_bytes()),
        )
    except BaseException as error:
        receipt.update(
            status="incomplete" if isinstance(error, TimeoutError) else "failed",
            error_type=type(error).__name__,
            error=str(error),
            traceback=traceback.format_exc(),
        )
        raise
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=0.5)
        try:
            receipt["hold_after"] = check_hold()
        except (OSError, ValueError) as error:
            receipt.update(hold_after=None, hold_error=str(error))
        receipt.update(
            hold_unchanged=receipt["hold_before"] is not None
            and receipt["hold_before"] == receipt.get("hold_after"),
            finished_at_utc=helpers.now(),
            total_supervised_process_seconds=time.monotonic() - started,
        )
        if not receipt["hold_unchanged"]:
            receipt["status"] = "failed"
        if receipt["total_supervised_process_seconds"] > SUPERVISOR_SECONDS:
            receipt.update(status="incomplete", supervisor_budget_exceeded=True)
        save(out / "execution.json", receipt)
        print(
            json.dumps(
                dict(
                    output=str(out),
                    status=receipt["status"],
                    elapsed_seconds=receipt["total_supervised_process_seconds"],
                ),
                ensure_ascii=False,
            )
        )
    if receipt["status"] != "passed":
        raise RuntimeError("El cierre no confirmó todas las condiciones del benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    options = parser.parse_args()
    if not options.run:
        if options.worker:
            parser.error("El hijo también necesita la activación explícita --run")
        print(json.dumps(protocol(), ensure_ascii=False, indent=2))
        return
    if options.output is None:
        parser.error("--run necesita una salida privada nueva mediante --output")
    root, out = options.repository.resolve(strict=True), options.output.resolve()
    if options.worker:
        if options.deadline is None:
            parser.error("El hijo necesita el plazo fijado por el supervisor")
        worker(root, out, options.deadline)
    else:
        supervise(root, out)


if __name__ == "__main__":
    main()
