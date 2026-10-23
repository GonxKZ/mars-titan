"""Medir ocho controles regionales sobre fixtures sin ajustar parámetros.

Sin --run imprime el protocolo sin importar Torch. Cada ejecución selecciona
un estrato FP32/FP64 y lote 1/256, con dos órdenes y prefijos propios. Reutiliza
la medición completa de prepare, sus comprobaciones y sus límites existentes.
"""

import argparse
import hashlib
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = "configs/experiments/regional-hierarchy-v1.json"
DTYPES, BATCHES = ("float32", "float64"), (1, 256)


def declaration(root):
    declared = json.loads((root / DECLARATION).read_text())
    recipe = json.loads((root / declared["base_recipe"]).read_text())
    return declared, recipe


def protocol(repository=None):
    root = ROOT if repository is None else Path(repository).resolve()
    declared, _ = declaration(root)
    cells = [
        dict(
            cell_id=f"{context or 'none'}/{arm}",
            variant="mac_online",
            context=context,
            arm=arm,
        )
        for context in (None, "market_memory_v1")
        for arm in declared["arms"]
    ]
    return dict(
        kind="pre_mac_regional_hierarchy_fixture_matrix_v1",
        cells=cells,
        arm_configurations=declared["arms"],
        dtypes=list(DTYPES),
        batches=list(BATCHES),
        condition_records=128,
        prefix_prepare_calls=256,
        timed_prepare_calls=896,
        total_prepare_calls=1152,
        warmup_per_record=2,
        repetitions_per_record=5,
        device="cuda:0",
        profile_overrides=dict(hidden_size=32, seed=42, max_batch=256, head="quantile_head_v1"),
        timed_scope="complete_prepare_and_CUDA_completion_sync",
        fixture="regional_program_cuda.batch_for_events_0_1_2_without_labels",
        state="own_two_event_prefix_per_condition_and_order",
        order="all_16_cells_then_exact_global_reverse",
        parameter_comparison="exact_values_names_shapes_within_context_and_dtype",
        geometry_comparison="derived_per_configuration_not_forced_equal_across_topologies",
        legacy_control="20_local_2_summary_different_schedule_no_isolated_hierarchy_claim",
        hierarchy_controls="20_local_10_auxiliary_receiver_evaluations_shared_schedule",
        statistics="five_raw_samples_min_median_max_no_tail_or_predictive_inference",
        tf32=False,
        autocast=False,
        deterministic_algorithms=True,
        allocator_cap_bytes=512 * 1024**2,
        free_headroom_bytes=256 * 1024**2,
        cpu_proof_cache_cap_bytes=128 * 1024**2,
        json_report_cap_bytes=16 * 1024**2,
        child_budget_seconds=180,
        optimizer_steps=0,
        scientific_evaluation=False,
        final_test_access=False,
        energy_measured=False,
        physical_transfers_measured=False,
    )


def execution_order(cells, pass_index):
    if type(pass_index) is not int or pass_index not in (0, 1):
        raise ValueError("La pasada debe ser el entero 0 o 1")
    names = [cell["cell_id"] for cell in cells]
    return names if pass_index == 0 else list(reversed(names))


def discard_geometry_proof(known):
    """Separar buffers derivados de topologías distintas de sus pesos compartidos."""
    for key in tuple(known):
        if key.startswith("result/buffer/regional_execution."):
            del known[key]


def verify_parameter_shapes(records):
    expected = None
    for record in records:
        actual = {
            name: shape
            for name, shape in record["parameter_shapes"].items()
            if not name.startswith("buffer/")
        }
        if expected is not None and actual != expected:
            raise AssertionError("Una condición cambia nombres o formas de los pesos compartidos")
        expected = actual


def configure_environment():
    settings = dict(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
    )
    os.environ.update(settings)
    return settings


def finalize_report(report, guard, core, helper, started, limit):
    """Conservar el error principal y registrar fallos de limpieza o de la guarda."""
    if guard is not None:
        try:
            guard.remove()
        except Exception as error:
            report.update(status="failed", cleanup_error=f"{type(error).__name__}: {error}")
    try:
        report["hold_sha256_final"] = core.check_hold()
    except Exception as error:
        report.update(status="failed", hold_error=f"{type(error).__name__}: {error}")
    report["hold_unchanged"] = (
        report.get("hold_before") == report.get("hold_sha256_final") == core.HOLD_SHA256
    )
    report.update(finished_at_utc=helper.now(), total_process_seconds=time.monotonic() - started)
    if report["total_process_seconds"] > limit:
        report["budget_exceeded"] = True
        if report["status"] != "failed":
            report["status"] = "incomplete"


def configurations(root, plan):
    from test_financial_adapter import specification

    from mars_titan.models.titans.financial import FinancialConfig

    _, recipe = declaration(root)
    common = dict(recipe["predictor"], **plan["profile_overrides"])
    common.pop("dtype")
    common["regional_fusion"] = dict(
        topology="self",
        routing="uniform",
        cycles=2,
        local_steps=2,
        summary=True,
        local_gain=0.25,
        summary_gain=0.25,
    )
    return {
        cell["cell_id"]: FinancialConfig(
            specification(),
            **common,
            variant="mac_online",
            session_context=cell["context"],
            regional_execution=plan["arm_configurations"][cell["arm"]],
        )
        for cell in plan["cells"]
    }


def run(root, out, dtype, batch):
    import importlib.util

    helper_path = root / "benchmarks/regional_program_matrix_cuda.py"
    spec = importlib.util.spec_from_file_location("_hierarchy_matrix_reused", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    plan = protocol(root)
    if out.exists() or out.is_relative_to(root):
        raise ValueError("La salida debe ser nueva y externa al repositorio")
    out.mkdir(parents=True)
    core = helper.load(root, "regional_program_cuda")
    h = core.helpers_on(root)
    report = dict(
        protocol=plan,
        execution_stratum=dict(dtype=dtype, batch=batch, expected_records=32),
        status="incomplete",
        started_at_utc=h.now(),
        records=[],
        prefix_calls_completed=0,
        timed_calls_completed=0,
        no_summary_pairs_checked=0,
    )
    guard, started = None, time.monotonic()
    deadline = started + plan["child_budget_seconds"]

    def alarm(signum, frame):
        raise TimeoutError("El estrato jerárquico agotó su límite interno")

    previous = signal.signal(signal.SIGALRM, alarm)
    signal.setitimer(signal.ITIMER_REAL, plan["child_budget_seconds"])

    def sources():
        pinned = core.manifest(root, h)
        for name in (
            "benchmarks/regional_program_matrix_cuda.py",
            str(Path(__file__).relative_to(root)),
        ):
            pinned[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
        return pinned

    try:
        report["hold_before"] = core.check_hold()
        pinned = sources()
        helper.save(out, "source-manifest.json", pinned)
        helper.save(out, "protocol.json", plan)
        report["numerical_environment"] = configure_environment()
        import torch

        configs = configurations(root, plan)
        if torch.cuda.is_initialized():
            raise AssertionError("La construcción de configuraciones inicializó CUDA")
        report["hardware_before"] = h.nvidia()
        if not torch.cuda.is_available():
            raise RuntimeError("Falta CUDA, sin sustitución por CPU")
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
        free, total = torch.cuda.mem_get_info("cuda:0")
        if free < h.ALLOCATOR_CAP + h.HEADROOM:
            raise MemoryError("Falta VRAM para el asignador y su margen")
        torch.cuda.set_per_process_memory_fraction(h.ALLOCATOR_CAP / total, "cuda:0")
        from mars_titan.training.learning_hold import install_optimizer_guard

        def blocked(reason):
            raise RuntimeError(reason)

        guard = install_optimizer_guard(blocked, path=core.HOLD)
        if guard is None:
            raise RuntimeError("La protección de optimizadores no está activa")
        report["environment"] = dict(
            python=sys.version,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            loadavg=os.getloadavg(),
            cpu_threads=2,
        )
        cells = {cell["cell_id"]: cell for cell in plan["cells"]}
        batches, refs, known, shape_groups = None, {}, {}, {}
        rng = (torch.random.get_rng_state().clone(), torch.cuda.get_rng_state(0).clone())
        for pass_index in (0, 1):
            for position, cell_id in enumerate(execution_order(plan["cells"], pass_index)):
                cell = cells[cell_id]
                discard_geometry_proof(known)
                record = dict(
                    cell_id=cell_id,
                    dtype=dtype,
                    batch=batch,
                    pass_index=pass_index,
                    position=position,
                    status="incomplete",
                    samples=[],
                    warmup_seconds=[],
                    financial_configuration=configs[cell_id].identity(),
                )
                report["records"].append(record)
                batches = helper.condition(
                    torch,
                    h,
                    core,
                    configs[cell_id],
                    cell,
                    record,
                    batches,
                    refs,
                    known,
                    shape_groups.setdefault(cell_id, {}),
                    {},
                    report,
                    deadline,
                )
                expected = dict(
                    local_node_evaluations=20 * batch,
                    summary_evaluations=(2 if cell["arm"].startswith("legacy_") else 10) * batch,
                )
                if record["observed_regional_work"] != expected:
                    raise AssertionError("El trabajo observado no conserva el calendario declarado")
                if not torch.equal(rng[0], torch.random.get_rng_state()) or not torch.equal(
                    rng[1], torch.cuda.get_rng_state(0)
                ):
                    raise AssertionError("Una condición consumió el RNG compartido")
                helper.save(out, "measurement.json", report)
                torch.cuda.empty_cache()
        for context in (None, "market_memory_v1"):
            verify_parameter_shapes(
                [row for row in report["records"] if cells[row["cell_id"]]["context"] == context]
            )
        if (
            len(report["records"]),
            report["prefix_calls_completed"],
            report["timed_calls_completed"],
        ) != (32, 64, 224):
            raise AssertionError("El estrato no completó sus cantidades previstas")
        if sources() != pinned:
            raise ValueError("Las fuentes cambiaron durante la medición")
        report.update(status="passed", source_manifest_unchanged=True, hardware_after=h.nvidia())
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
        finalize_report(report, guard, core, h, started, plan["child_budget_seconds"])
        helper.save(out, "measurement.json", report)
    if report["status"] != "passed":
        raise RuntimeError("El estrato no conserva un cierre técnico completo")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=DTYPES)
    parser.add_argument("--batch", type=int, choices=BATCHES)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(protocol(), ensure_ascii=False, indent=2))
        return
    if args.dtype is None or args.batch is None or args.output is None:
        parser.error("--run requiere --dtype, --batch y --output")
    for part in ("src", ".", "tests/models/titans", "tests/memory"):
        sys.path.insert(0, str(ROOT / part))
    run(ROOT, args.output.resolve(), args.dtype, args.batch)


if __name__ == "__main__":
    main()
