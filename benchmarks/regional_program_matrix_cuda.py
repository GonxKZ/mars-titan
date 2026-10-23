"""Matriz técnica de modos de memoria, contexto y programas, sin aprendizaje.

Sin --run imprime el protocolo sin importar Torch ni crear una salida. El modo
activo exige un estrato --dtype/--batch y valida las 48 configuraciones en CPU
antes de inicializar CUDA. Mide las 26 admitidas en ambos órdenes con una sola
instancia GPU residente y conserva los rechazos.
El supervisor exterior corresponde al llamante. No compara cuatro codificadores.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import signal
import statistics
import sys
import time
import traceback
from pathlib import Path

MODES = ("transformer_direct", "mac_disabled", "mac_frozen", "mac_online")
CONTEXTS = (None, "market_memory_v1")
DTYPES, BATCHES = ("float32", "float64"), (1, 256)
CPU_CAP, JSON_CAP = 128 * 1024**2, 16 * 1024**2
ROOT = Path(__file__).resolve().parents[1]


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def declaration(root):
    source = root / "configs/experiments/regional-program-v1.json"
    declared = json.loads(source.read_text())
    base = root / declared["base_recipe"]
    return (
        declared,
        json.loads(base.read_text()),
        {
            str(source.relative_to(root)): digest(source.read_bytes()),
            str(base.relative_to(root)): digest(base.read_bytes()),
        },
    )


def protocol(repository=None):
    root = ROOT if repository is None else Path(repository).resolve()
    declared, _, hashes = declaration(root)
    cells = []
    for mode in MODES:
        for context in CONTEXTS:
            group = f"{mode}/{context or 'none'}"
            for arm in declared["arms"]:
                rejection = (
                    "context_requires_mac_online"
                    if context and mode != "mac_online"
                    else "contextual_program_requires_context"
                    if arm == "contextual_norm" and context is None
                    else None
                )
                cells.append(
                    dict(
                        cell_id=f"{group}/{arm}",
                        variant=mode,
                        context=context,
                        arm=arm,
                        baseline_cell_id=f"{group}/legacy_post_fusion",
                        expected_admissible=rejection is None,
                        expected_rejection=rejection,
                    )
                )
    return dict(
        kind="regional_program_memory_mode_matrix_fixture_cost_v1",
        cells=cells,
        planned_cells=48,
        admissible_cells=26,
        rejected_cells=22,
        condition_records=208,
        prefix_prepare_calls=416,
        timed_prepare_calls=1456,
        total_prepare_calls=1872,
        warmup_per_record=2,
        repetitions_per_record=5,
        dtypes=list(DTYPES),
        batches=list(BATCHES),
        profile_overrides=dict(hidden_size=32, seed=42, max_batch=256, head="quantile_head_v1"),
        scientific_declarations_changed=False,
        declaration_sha256=hashes,
        script_sha256=digest(Path(__file__).read_bytes()),
        eligibility="planned_until_real_FinancialConfig_constructor_checked_before_CUDA",
        order="all_26_admissible_cells_then_exact_global_reverse_per_dtype_and_batch",
        predictor_residency="one_GPU_model_per_condition_initialization_excluded_from_prepare_time",
        fixture="test_context_selection.batch_on_via_regional_program_cuda.batch_for",
        fixture_events=dict(prefix=[0, 1], timed=2, observation_ordinal=3),
        timed_scope="complete_FinancialPredictor.prepare_plus_completion_sync",
        prefix_scope="two_own_prepare_and_context_commits_rebuilt_per_order",
        shared_parameter_check="same_names_shapes_and_exact_values_within_dtype_including_buffers",
        budgets="report_parameter_extras_across_MAC_and_PT4_no_equal_architectural_budget_claim",
        no_summary_parity="all_result_tensor_leaves_except_regional_trace_exact_within_mode_context_dtype_batch",
        no_summary_pairs=40,
        temporal_policy="same_observations_masks_and_cuts_no_labels",
        summary="five_samples_min_median_max_no_p95_p99_or_confidence_interval",
        interpretation="four_memory_modes_with_one_common_fixture_encoder",
        comparisons="fixed_end_vs_interleaved_isolates_schedule_legacy_PT4_vs_program_changes_query_position_too",
        device="cuda:0",
        cpu_threads=2,
        interop_threads=1,
        tf32=False,
        autocast=False,
        deterministic_algorithms=True,
        mha_fastpath=False,
        child_budget_seconds=240,
        external_job_budget_seconds=300,
        external_phase_budget_seconds=1500,
        allocator_cap_bytes=512 * 1024**2,
        free_vram_headroom_bytes=256 * 1024**2,
        cpu_proof_tensor_cap_bytes=CPU_CAP,
        report_cap_bytes=JSON_CAP,
        optimizer_steps=0,
        scientific_evaluation=False,
        final_test_access=False,
        energy_measured=False,
        physical_transfers_measured=False,
    )


def execution_scope(plan, dtype=None, batch=None):
    if (dtype is None) != (batch is None):
        raise ValueError("El estrato necesita --dtype y --batch juntos")
    if dtype is not None and (type(dtype) is not str or dtype not in plan["dtypes"]):
        raise ValueError("El dtype del estrato debe ser float32 o float64")
    if batch is not None and (type(batch) is not int or batch not in plan["batches"]):
        raise ValueError("El lote del estrato debe ser 1 o 256")
    dtypes = list(plan["dtypes"]) if dtype is None else [dtype]
    batches = list(plan["batches"]) if batch is None else [batch]
    global_strata = len(plan["dtypes"]) * len(plan["batches"])
    return dict(
        dtypes=dtypes,
        batches=batches,
        **{
            name: plan[name] * len(dtypes) * len(batches) // global_strata
            for name in (
                "condition_records",
                "prefix_prepare_calls",
                "timed_prepare_calls",
                "total_prepare_calls",
                "no_summary_pairs",
            )
        },
    )


def execution_order(cells, pass_index):
    if type(pass_index) is not int or pass_index not in (0, 1):
        raise ValueError("La pasada debe ser 0 o 1")
    admitted = [cell["cell_id"] for cell in cells if cell["expected_admissible"]]
    return admitted if pass_index == 0 else list(reversed(admitted))


def load(root, name):
    spec = importlib.util.spec_from_file_location(f"_matrix_{name}", root / f"benchmarks/{name}.py")
    if spec is None or spec.loader is None:
        raise ImportError("Falta el helper del benchmark")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(out, name, value):
    encoded = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    if len(encoded.encode()) > JSON_CAP:
        raise MemoryError("El informe supera 16 MiB")
    temporary = out / f"{name}.tmp"
    temporary.write_text(encoded + "\n")
    temporary.replace(out / name)


def admission(root, plan):
    from test_financial_adapter import specification

    from mars_titan.models.titans.financial import FinancialConfig

    declared, base, _ = declaration(root)
    common = dict(base["predictor"], **plan["profile_overrides"])
    common.pop("dtype")
    inputs, configs, observed = specification(), {}, []
    for cell in plan["cells"]:
        options = dict(common, variant=cell["variant"], session_context=cell["context"])
        options.update(declared["arms"][cell["arm"]]["predictor_changes"])
        try:
            config, reason = FinancialConfig(inputs, **options), None
        except ValueError as error:
            config, reason = None, str(error)
        if (config is not None) != cell["expected_admissible"]:
            raise AssertionError(f"Cambió la admisibilidad de {cell['cell_id']}: {reason}")
        observed.append(dict(cell, admissible=config is not None, reason=reason))
        if config is not None:
            configs[cell["cell_id"]] = config
    if len(configs) != 26 or len(observed) != 48:
        raise AssertionError("La matriz no conserva sus 26 configuraciones admitidas")
    return configs, observed


def cache_bytes(*mappings):
    return sum(
        value.numel() * value.element_size() for mapping in mappings for value in mapping.values()
    )


def condition(
    torch, h, core, config, cell, record, batches, refs, known, groups, pair, report, deadline
):
    from mars_titan.models.titans.financial import FinancialPredictor

    h.check_deadline(deadline)
    start = time.perf_counter()
    model = FinancialPredictor(config, device="cuda:0", dtype=getattr(torch, record["dtype"]))
    weights = model.state_dict()
    if model.session_context is not None:
        weights["session_context.output"] = (
            torch.eye(32, device="cuda:0", dtype=model.head.weight.dtype) * 0.2
        )
        model.load_state_dict(weights)
    model.eval().requires_grad_(False)
    torch.cuda.synchronize("cuda:0")
    record["initialization_seconds"] = time.perf_counter() - start
    if batches is None:
        start = time.perf_counter()
        flows = tuple(f"US/F{i:04d}" for i in range(record["batch"]))
        batches = [core.batch_for(torch, model, flows, event) for event in range(3)]
        record["input_initialization_seconds"] = time.perf_counter() - start
    batch, prefixes = batches[2], batches[:2]
    payload = [
        {name: getattr(value, name) for name in value.__dataclass_fields__ if name != "_versions"}
        for value in batches
    ]
    input_sha = h.proof(torch, payload)[0]
    record["input_sha256"] = input_sha
    named = dict(model.named_parameters()) | {
        f"buffer/{name}": value for name, value in model.named_buffers()
    }
    parameter_sha, _, cpu = h.proof(torch, named, keep=True)
    common = known.keys() & cpu.keys()
    h.strict_equal(
        torch, {name: known[name] for name in common}, {name: cpu[name] for name in common}
    )
    group = (cell["variant"], cell["context"])
    if groups.setdefault(group, set(cpu)) != set(cpu):
        raise AssertionError("Un programa cambió los parámetros de su grupo")
    known.update({name: value for name, value in cpu.items() if name not in known})
    size = cache_bytes(known, cpu, *pair.values())
    if size > CPU_CAP:
        raise MemoryError("La caché de pruebas CPU supera 128 MiB")
    report["cpu_proof_tensor_peak_bytes"] = max(report.get("cpu_proof_tensor_peak_bytes", 0), size)
    record.update(
        parameter_sha256=parameter_sha,
        parameter_shapes={name: list(value.shape) for name, value in named.items()},
        parameter_elements=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        mac_module_present=model.mac is not None,
        associative_update_enabled=cell["variant"] == "mac_online",
        context_present=model.session_context is not None,
    )
    del cpu, named, weights
    prefix_started = time.perf_counter()
    state = model.initial_state(batch.flow_ids)
    context = model.initial_context() if model.session_context is not None else None
    watched = {FinancialPredictor.prepare.__code__: "core"}
    if model.mac is not None:
        watched[type(model.mac).forward.__code__] = "mac"
    calls = dict(core=0, mac=0)

    def observe(frame, event, result):
        if event == "call" and frame.f_code in watched:
            calls[watched[frame.f_code]] += 1

    previous = sys.getprofile()
    sys.setprofile(observe)
    try:
        for prefix in prefixes:
            h.check_memory(torch)
            result = model.prepare(
                prefix, state, **(dict(context=context) if context is not None else {})
            )
            h.check_memory(torch)
            state = result.next_state
            context = model.commit_context(result.next_context) if context is not None else None
            report["prefix_calls_completed"] += 1
    finally:
        sys.setprofile(previous)
    del result
    torch.cuda.synchronize("cuda:0")
    record["prefix_seconds"] = time.perf_counter() - prefix_started
    if not bool((state.observed_steps == 2).all()):
        raise AssertionError("El prefijo no conserva exactamente dos observaciones por flujo")
    if calls != dict(core=2, mac=2 if model.mac is not None else 0):
        raise AssertionError("El prefijo no realizó una preparación y un MAC admisible por evento")
    initial_sha = h.proof(torch, dict(state=state, context=context))[0]
    model_sha = h.proof(torch, model.state_dict())[0]
    record.update(
        initial_state_sha256=initial_sha, observed_prefix_calls=calls, model_state_sha256=model_sha
    )
    for iteration in range(7):
        h.check_deadline(deadline)
        h.check_memory(torch)
        torch.cuda.synchronize("cuda:0")
        torch.cuda.reset_peak_memory_stats("cuda:0")
        start = time.perf_counter()
        result = model.prepare(
            batch, state, **(dict(context=context) if context is not None else {})
        )
        torch.cuda.synchronize("cuda:0")
        elapsed = time.perf_counter() - start
        memory = h.check_memory(torch)
        sha, _, retained = h.proof(
            torch, result, keep=iteration == 0 and cell["arm"].endswith("no_summary")
        )
        if iteration == 0 and model.regional_execution is not None:
            rows = model.regional_execution.export_trace(
                result.regional_trace, model.regional_fusion
            )
            record["observed_regional_work"] = model.regional_execution.verify_exported_traces(
                rows, model.regional_fusion
            )
            record["selected_program_rows"] = {
                name: sum(row["program"] == name for row in rows)
                for name in sorted({row["program"] for row in rows})
            }
        snapshot = (parameter_sha, initial_sha, sha)
        if refs.setdefault(cell["cell_id"], snapshot) != snapshot:
            raise AssertionError("La reconstrucción, una repetición o el orden cambió el resultado")
        if not torch.equal(result.next_state.observed_steps, state.observed_steps + 1):
            raise AssertionError("La observación global no avanzó exactamente una vez")
        expected = 3 if cell["variant"] == "mac_online" else 0
        if result.next_state.mac is not None and not bool(
            (result.next_state.mac.memory.steps == expected).all()
        ):
            raise AssertionError("Los pesos rápidos no conservan los pasos permitidos por el modo")
        if retained:
            tensors = {
                name: value
                for name, value in retained.items()
                if isinstance(value, torch.Tensor) and not name.startswith("result/regional_trace/")
            }
            size = cache_bytes(known, tensors, *pair.values())
            if size > CPU_CAP:
                raise MemoryError("La pareja sin resumen supera la caché CPU")
            report["cpu_proof_tensor_peak_bytes"] = max(report["cpu_proof_tensor_peak_bytes"], size)
            if group in pair:
                h.strict_equal(torch, pair.pop(group), tensors)
                report["no_summary_pairs_checked"] += 1
            else:
                pair[group] = tensors
        if iteration < 2:
            record["warmup_seconds"].append(elapsed)
        else:
            record["samples"].append(dict(seconds=elapsed, **memory, **h.ram()))
        report["timed_calls_completed"] += 1
        del result, retained
    if (
        h.proof(torch, dict(state=state, context=context))[0] != initial_sha
        or h.proof(torch, model.state_dict())[0] != model_sha
    ):
        raise AssertionError("prepare modificó el prefijo o los parámetros compartidos")
    model.verify_parameter_identity()
    if h.proof(torch, payload)[0] != input_sha or any(
        parameter.grad is not None for parameter in model.parameters()
    ):
        raise AssertionError("La condición cambió los inputs o acumuló gradientes compartidos")
    seconds = [row["seconds"] for row in record["samples"]]
    record.update(
        status="passed",
        min_ms=1000 * min(seconds),
        median_ms=1000 * statistics.median(seconds),
        max_ms=1000 * max(seconds),
        rows_per_second=len(batch.flow_ids) / statistics.mean(seconds),
    )
    return batches


def run(root, out, dtype, batch):
    plan = protocol(root)
    scope = execution_scope(plan, dtype, batch)
    if dtype is None:
        raise ValueError("La ejecución necesita un estrato --dtype/--batch explícito")
    if out.exists() or out.is_relative_to(root):
        raise ValueError("La salida debe ser nueva y externa al repositorio")
    out.mkdir(parents=True)
    core = load(root, "regional_program_cuda")
    h = core.helpers_on(root)
    report = dict(
        protocol=plan,
        execution_stratum=scope,
        status="incomplete",
        started_at_utc=h.now(),
        records=[],
        prefix_calls_completed=0,
        timed_calls_completed=0,
        no_summary_pairs_checked=0,
    )
    started = time.monotonic()
    budget = plan["child_budget_seconds"]
    deadline = started + budget
    guard = None

    def alarm(signum, frame):
        raise TimeoutError(f"El estrato agotó sus {budget} segundos internos")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, budget)
    try:
        report["hold_before"] = core.check_hold()
        save(out, "protocol.json", plan)
        save(out, "execution-stratum.json", scope)
        pinned = core.manifest(root, h)
        name = "benchmarks/regional_program_matrix_cuda.py"
        pinned[name] = digest((root / name).read_bytes())
        if pinned[name] != plan["script_sha256"]:
            raise ValueError("La copia no coincide con la fuente de la matriz")
        save(out, "source-manifest.json", pinned)
        os.environ.update(
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            PYTHONDONTWRITEBYTECODE="1",
        )
        import torch

        configs, report["eligibility"] = admission(root, plan)
        if torch.cuda.is_initialized():
            raise AssertionError("La validación de configuraciones inicializó CUDA")
        save(out, "measurement.json", report)
        report["hardware_before"] = h.nvidia()
        if not torch.cuda.is_available():
            raise RuntimeError("Falta CUDA. No se permite sustitución por CPU")
        torch.cuda.set_device("cuda:0")
        if "4070" not in torch.cuda.get_device_name(0):
            raise RuntimeError("cuda:0 no es la RTX 4070 prevista")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.mha.set_fastpath_enabled(False)
        if torch.is_autocast_enabled("cuda") or torch.is_inference_mode_enabled():
            raise ValueError("La matriz exige precisión explícita fuera de inference_mode")
        free, total = torch.cuda.mem_get_info("cuda:0")
        if free < h.ALLOCATOR_CAP + h.HEADROOM:
            raise MemoryError("Falta reserva para 512 MiB de asignador y 256 MiB libres")
        torch.cuda.set_per_process_memory_fraction(h.ALLOCATOR_CAP / total, "cuda:0")
        from mars_titan.training.learning_hold import install_optimizer_guard

        def blocked(reason):
            raise RuntimeError(reason)

        guard = install_optimizer_guard(blocked, path=core.HOLD)
        if guard is None:
            raise RuntimeError("La protección de optimizadores no está activa")
        cells = {cell["cell_id"]: cell for cell in plan["cells"]}
        report["environment"] = dict(
            python=sys.version,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            cpu_count=os.cpu_count(),
            loadavg=os.getloadavg(),
        )
        for dtype_name in scope["dtypes"]:
            for batch_size in scope["batches"]:
                batches, refs, known, groups = None, {}, {}, {}
                rng = (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone())
                for pass_index in (0, 1):
                    pair = {}
                    for position, cell_id in enumerate(execution_order(plan["cells"], pass_index)):
                        config, cell = configs[cell_id], cells[cell_id]
                        record = dict(
                            cell_id=cell_id,
                            dtype=dtype_name,
                            batch=batch_size,
                            pass_index=pass_index,
                            position=position,
                            status="incomplete",
                            samples=[],
                            warmup_seconds=[],
                        )
                        report["records"].append(record)
                        batches = condition(
                            torch,
                            h,
                            core,
                            config,
                            cell,
                            record,
                            batches,
                            refs,
                            known,
                            groups,
                            pair,
                            report,
                            deadline,
                        )
                        if not torch.equal(rng[0], torch.random.get_rng_state()) or not torch.equal(
                            rng[1], torch.cuda.get_rng_state(0)
                        ):
                            raise AssertionError("La matriz consumió el RNG compartido")
                        save(out, "measurement.json", report)
                        torch.cuda.empty_cache()
                    if pair:
                        raise AssertionError("Falta contrastar una pareja sin resumen")
                del batches, known
                torch.cuda.empty_cache()
        if (
            len(report["records"]),
            report["prefix_calls_completed"],
            report["timed_calls_completed"],
        ) != (
            scope["condition_records"],
            scope["prefix_prepare_calls"],
            scope["timed_prepare_calls"],
        ) or report["no_summary_pairs_checked"] != scope["no_summary_pairs"]:
            raise AssertionError("El estrato no completó las cantidades declaradas")
        for dtype_name in scope["dtypes"]:
            records = [row for row in report["records"] if row["dtype"] == dtype_name]
            shared = set.intersection(*(set(row["parameter_shapes"]) for row in records))
            shared = {key for key in shared if not key.startswith("buffer/")}
            for row in records:
                row["shared_parameter_elements"] = sum(
                    math.prod(row["parameter_shapes"][key]) for key in shared
                )
                row["extra_parameter_elements"] = (
                    row["parameter_elements"] - row["shared_parameter_elements"]
                )
                row["extra_parameter_names"] = sorted(
                    set(row["parameter_shapes"])
                    - shared
                    - {key for key in row["parameter_shapes"] if key.startswith("buffer/")}
                )
        current = core.manifest(root, h)
        current[name] = digest((root / name).read_bytes())
        if current != pinned:
            raise ValueError("Las fuentes cambiaron durante la matriz")
        report.update(
            status="passed",
            hold_sha256_final=core.check_hold(),
            source_manifest_unchanged=True,
            hardware_after=h.nvidia(),
        )
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
        if guard is not None:
            guard.remove()
        try:
            report["hold_sha256_final"] = core.check_hold()
        except (OSError, ValueError) as error:
            report.update(status="failed", hold_error=str(error))
        report["hold_unchanged"] = (
            report.get("hold_before") == report.get("hold_sha256_final") == core.HOLD_SHA256
        )
        report.update(
            finished_at_utc=h.now(),
            total_process_seconds=time.monotonic() - started,
            ram_final=h.ram(),
            loadavg_final=os.getloadavg(),
        )
        if report["total_process_seconds"] > budget:
            report.update(status="incomplete", budget_exceeded=True)
        save(out, "measurement.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=DTYPES)
    parser.add_argument("--batch", type=int, choices=BATCHES)
    args = parser.parse_args()
    root = args.repository.resolve(strict=True)
    plan = protocol(root)
    try:
        scope = execution_scope(plan, args.dtype, args.batch)
    except ValueError as error:
        parser.error(str(error))
    if not args.run:
        if args.dtype is not None:
            plan["execution_stratum"] = scope
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if args.dtype is None:
        parser.error("--run necesita --dtype y --batch para seleccionar un estrato")
    if args.output is None:
        parser.error("--run necesita una salida externa nueva")
    for part in ("src", ".", "tests/models/titans", "tests/memory"):
        sys.path.insert(0, str(root / part))
    run(root, args.output.resolve(), args.dtype, args.batch)


if __name__ == "__main__":
    main()
