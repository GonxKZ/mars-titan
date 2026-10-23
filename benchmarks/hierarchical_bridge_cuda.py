"""Coste técnico del puente jerárquico y de prepare, con fixtures y sin aprendizaje.

Sin --run imprime el protocolo sin importar Torch. Ejecutar una copia privada con
el supervisor local y --repository. No mide utilidad predictiva ni el corpus real.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import fields, replace
from pathlib import Path

VARIANTS = ("flat-j4", "tree-l1", "tree-l2", "tree-l4", "star-l2", "feedback-off-l2")
SHAPES = ((1, 8), (1, 1024), (256, 8), (256, 1024))
WARMUP, REPETITIONS = 3, 10
HOLD = Path.home() / ".local/state/mars-titan/training-hold-2000.json"
HOLD_SHA = "8a658080c1ac4e63847eba58b177020339da6ec47021cf26d2701830073c87fd"


def protocol():
    return dict(
        kind="hierarchical_bridge_fixture_cost_v1",
        variants=VARIANTS,
        shapes=SHAPES,
        distinct_conditions_per_stage=len(VARIANTS) * len(SHAPES),
        records_per_stage=2 * len(VARIANTS) * len(SHAPES),
        logical_invocations_per_stage=2 * len(VARIANTS) * len(SHAPES) * (WARMUP + REPETITIONS),
        actual_prepare_calls=2
        * len(VARIANTS)
        * sum(math.ceil(b / 128) for b, _ in SHAPES)
        * (WARMUP + REPETITIONS),
        warmup=WARMUP,
        repetitions=REPETITIONS,
        orders=("normal", "reverse"),
        stages=("SharedEvidenceBridge.forward_FP32", "FrozenFinancialConsumer.prepare_FP64_blocks"),
        prepare_physical_block_rows=128,
        prepare_scope="all_logical_rows_in_sequential_prepare_calls_same_blocks_for_every_variant",
        prepare_merge="results_retained_as_tuple_no_output_merge_inside_timing",
        device="cuda:0",
        regions=4,
        hidden_size=32,
        neighbors=8,
        K=1,
        shared_parameters=True,
        rounds=2,
        hierarchy_local_steps=(1, 2, 4),
        flat_steps=4,
        same_compute_budget=False,
        new_updates_per_read=dict(zip(VARIANTS, (16, 18, 26, 42, 26, 26), strict=True)),
        includes="candidate_selection_and_readout_or_complete_prepare_plus_completion_sync",
        excludes="fixture_creation_initial_transfers_corpus_encoding_collection_persistence_feedback",
        cpu_reference="FP32_parameters_and_inputs_lifted_to_FP64_for_component",
        cpu_reference_block_rows=128,
        reference_blocking="all_rows_same_parameters_outside_timing_respecting_32MiB_guard",
        component_atol=2e-6,
        component_rtol=1e-5,
        feedback_off_vs_flat_tolerances=dict(
            component=dict(atol=2e-6, rtol=1e-5), prepare=dict(atol=1e-12, rtol=1e-12)
        ),
        percentiles="descriptive_linear_interpolation_ten_samples_not_stable_tail_estimates",
        maximum_process_seconds=180,
        allocator_limit_bytes=512 * 1024**2,
        minimum_free_vram_bytes=256 * 1024**2,
        optimizer_steps=0,
        scientific_evaluation=False,
        real_corpus=False,
        predictive_improvement_measured=False,
        energy_measured=False,
        physical_transfers_measured=False,
    )


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_helpers(root):
    path = root / "benchmarks/regional_decision_cuda.py"
    spec = importlib.util.spec_from_file_location("decision_cost_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(root, output):
    if output.exists() or not output.parent.is_dir() or output.is_relative_to(root):
        raise ValueError("La salida técnica debe ser nueva y quedar fuera del repositorio")
    if Path(__file__).resolve().is_relative_to(root):
        raise ValueError("Ejecutar una copia privada del guion, fuera del repositorio")
    if digest(HOLD) != HOLD_SHA:
        raise ValueError("La guarda de aprendizaje no corresponde al protocolo")
    start = time.monotonic()
    helpers = load_helpers(root)
    pinned = helpers.manifest(root)
    for path in (root, root / "src", root / "tests/models/titans", root / "tests/memory"):
        sys.path.insert(0, str(path))
    import torch
    from test_bridge_episodic_readout import configuration
    from test_context_selection import batch_on
    from test_episodic_readout import CODEC, CONTEXT, source
    from test_market_memory import pt4

    from mars_titan.models.titans.bridge_hierarchy import BridgeHierarchyConfig
    from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
    from mars_titan.models.titans.episodic_readout import EpisodicReadout
    from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
    from mars_titan.models.titans.evidence_bridge import SharedEvidenceBridge
    from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA no está disponible, sin sustitución por CPU")
    torch.cuda.set_device("cuda:0")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.mha.set_fastpath_enabled(False)
    torch.use_deterministic_algorithms(True)
    report = dict(
        protocol=protocol(),
        source_sha256=pinned,
        script_sha256=digest(Path(__file__)),
        helper_sha256=digest(root / "benchmarks/regional_decision_cuda.py"),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        records=[],
        status="incomplete",
        nvidia_smi=subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.used,driver_version", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip(),
    )

    def save():
        temporary = output.with_suffix(".temporary")
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        temporary.replace(output)

    def deadline():
        if time.monotonic() - start > 170:
            raise TimeoutError("La medida técnica agotó su límite interno")

    model = pt4(None).to("cuda:0").eval().requires_grad_(False)
    engines, components, references, paired = {}, {}, {}, {}
    for variant in VARIANTS:
        hierarchy = (
            None
            if variant == "flat-j4"
            else BridgeHierarchyConfig(
                topology="star" if variant == "star-l2" else "tree",
                local_steps=4 if variant == "tree-l4" else 1 if variant == "tree-l1" else 2,
                feedback=variant != "feedback-off-l2",
                max_node_updates_per_row=1596,
            )
        )
        base = configuration()
        config = replace(
            base,
            evidence_bridge=replace(
                base.evidence_bridge,
                regions=("v0", "v1", "v2", "v3"),
                topology="self",
                cycles=0,
                recurrence=BridgeRecurrenceConfig(4, 1216) if hierarchy is None else None,
                hierarchy=hierarchy,
            ),
        )
        reader = (
            EpisodicReadout(config, device="cuda:0", dtype=torch.float64)
            .eval()
            .requires_grad_(False)
        )
        engines[variant] = FrozenFinancialConsumer(model, readout=reader)
        component = (
            SharedEvidenceBridge(config.evidence_bridge, device="cuda:0", dtype=torch.float32)
            .eval()
            .requires_grad_(False)
        )
        reference = (
            SharedEvidenceBridge(config.evidence_bridge, dtype=torch.float64)
            .eval()
            .requires_grad_(False)
        )
        with torch.no_grad():
            for name, value in reference.named_parameters():
                value.copy_(dict(component.named_parameters())[name].cpu().double())
        components[variant], references[variant] = component, reference
        for stage, module in (("prepare", reader), ("component", component)):
            fingerprints = {
                name: helpers.proof(torch, value)[0] for name, value in module.named_parameters()
            }
            if stage not in paired:
                paired[stage] = fingerprints
            if paired[stage] != fingerprints:
                raise AssertionError("Los controles no conservan parámetros emparejados")
    report["parameter_sha256"] = paired
    report["model_ids"] = {name: engine.model_id for name, engine in engines.items()}
    report["public_script_sha256"] = digest(root / "benchmarks/hierarchical_bridge_cuda.py")
    if report["public_script_sha256"] != report["script_sha256"]:
        raise ValueError("La copia del guion difiere de su fuente pública")
    report["resident_tensor_bytes"] = {
        stage: {
            name: sum(
                value.numel() * value.element_size()
                for value in (*module.parameters(), *module.buffers())
            )
            for name, module in modules.items()
        }
        for stage, modules in (
            ("component", components),
            ("reference_cpu", references),
            ("readout", {name: engine.readout for name, engine in engines.items()}),
        )
    }
    report["cases"] = []
    report["common_predictor_tensor_bytes"] = sum(
        value.numel() * value.element_size() for value in (*model.parameters(), *model.buffers())
    )
    save()
    with torch.no_grad():
        for batch_size, episodes in SHAPES:
            deadline()
            flows = tuple(f"US/F{i:04d}" for i in range(batch_size))
            batch = batch_on(model, flows, 0)
            initial = model.initial_state(flows)
            physical = [
                (
                    batch.select(tuple(range(at, min(at + 128, batch_size)))),
                    model.select_state(initial, flows[at : at + 128]),
                )
                for at in range(0, batch_size, 128)
            ]
            raw = source(episodes)
            raw["keys"] = raw["keys"].repeat((math.ceil(episodes / len(raw["keys"])), 1))[:episodes]
            raw["labels"] = raw["labels"].float().double()
            snapshots = {
                device: EpisodeSnapshot.create(
                    **raw,
                    codec_id=CODEC,
                    context_id=CONTEXT,
                    cutoff=batch.prediction_at[0],
                    device=device,
                    dtype=dtype,
                )
                for device, dtype in (("cuda:0", torch.float32), ("cpu", torch.float64))
            }
            financial_snapshot = EpisodeSnapshot.create(
                **raw,
                codec_id=CODEC,
                context_id=CONTEXT,
                cutoff=batch.prediction_at[0],
                device="cuda:0",
                dtype=torch.float64,
            )
            query = torch.linspace(
                -0.5, 0.5, batch_size * 32, device="cuda:0", dtype=torch.float32
            ).reshape(batch_size, 32)
            batch_payload = {
                field.name: getattr(batch, field.name)
                for field in fields(batch)
                if field.name != "_versions"
            }
            initial_proof, initial_bytes, _ = helpers.proof(torch, initial)
            batch_proof, batch_bytes, _ = helpers.proof(torch, batch_payload)
            snapshot_proof = helpers.proof(torch, financial_snapshot.export_cpu())[0]
            rng = (torch.get_rng_state(), torch.cuda.get_rng_state(0))
            report["cases"].append(
                dict(
                    batch=batch_size,
                    episodes=episodes,
                    initial_sha256=initial_proof,
                    batch_sha256=batch_proof,
                    snapshot_sha256=snapshot_proof,
                    initial_tensor_bytes=initial_bytes,
                    batch_tensor_bytes=batch_bytes,
                    query_tensor_bytes=query.numel() * query.element_size(),
                    component_snapshot_tensor_bytes=snapshots["cuda:0"].usage(),
                    prepare_snapshot_tensor_bytes=financial_snapshot.usage(),
                    physical_prepare_batches=[len(part.flow_ids) for part, _ in physical],
                )
            )
            for stage in ("component", "prepare"):
                expected_ids, stable, outputs = None, {}, {}
                for reverse in (False, True):
                    for variant in reversed(VARIANTS) if reverse else VARIANTS:
                        deadline()
                        samples, peak_extra, reference_errors = [], 0, {}
                        for iteration in range(WARMUP + REPETITIONS):
                            helpers.check_memory(torch)
                            torch.cuda.synchronize("cuda:0")
                            torch.cuda.reset_peak_memory_stats("cuda:0")
                            before = torch.cuda.memory_allocated("cuda:0")
                            t0 = time.perf_counter()
                            result = (
                                components[variant](
                                    query,
                                    snapshots["cuda:0"],
                                    context_id=CONTEXT,
                                    cutoff=batch.prediction_at[0],
                                )
                                if stage == "component"
                                else tuple(
                                    engines[variant].prepare(
                                        part, state, context_id=CONTEXT, snapshot=financial_snapshot
                                    )
                                    for part, state in physical
                                )
                            )
                            torch.cuda.synchronize("cuda:0")
                            elapsed = time.perf_counter() - t0
                            memory = helpers.check_memory(torch)
                            peak_extra = max(peak_extra, memory["peak_allocated_bytes"] - before)
                            proof, result_bytes, _ = helpers.proof(torch, result)
                            if variant in stable and stable[variant] != proof:
                                raise AssertionError("Una repetición cambió su resultado completo")
                            stable[variant] = proof
                            read = (
                                [result]
                                if stage == "component"
                                else [part.readout.reads[0].bridge for part in result]
                            )
                            ids = torch.cat([part.evidence_ids for part in read]).cpu()
                            if expected_ids is None:
                                expected_ids = ids
                            if not torch.equal(ids, expected_ids):
                                raise AssertionError("Los controles cambiaron los candidatos")
                            if stage == "component" and iteration == 0:
                                expected_blocks = [
                                    references[variant](
                                        block,
                                        snapshots["cpu"],
                                        context_id=CONTEXT,
                                        cutoff=batch.prediction_at[0],
                                    )
                                    for block in query.cpu().double().split(128)
                                ]
                                for field in ("evidence_ids", "selected_positions"):
                                    if not torch.equal(
                                        getattr(result, field).cpu(),
                                        torch.cat(
                                            [getattr(part, field) for part in expected_blocks]
                                        ),
                                    ):
                                        raise AssertionError(
                                            "La referencia FP64 cambió candidatos o posiciones"
                                        )
                                check_fields = [
                                    "values",
                                    "region_values",
                                    "evidence_weights",
                                    "bridge_weights",
                                ]
                                if variant != "flat-j4":
                                    check_fields += [
                                        "group_values",
                                        "root_values",
                                        "hierarchy_weights",
                                    ]
                                for field in check_fields:
                                    actual_value = getattr(result, field).cpu().double()
                                    expected_value = torch.cat(
                                        [getattr(part, field) for part in expected_blocks]
                                    )
                                    torch.testing.assert_close(
                                        actual_value,
                                        expected_value,
                                        atol=2e-6,
                                        rtol=1e-5,
                                    )
                                    reference_errors[field] = float(
                                        (actual_value - expected_value).abs().max()
                                    )
                            outputs[variant] = (
                                torch.cat([part.region_values for part in read]).detach().clone()
                            )
                            if iteration >= WARMUP:
                                samples.append(elapsed)
                            del result, read
                        report["records"].append(
                            dict(
                                stage=stage,
                                batch=batch_size,
                                episodes=episodes,
                                variant=variant,
                                reverse=reverse,
                                samples_s=samples,
                                p50_ms=1000 * statistics.median(samples),
                                p95_ms=1000 * helpers.percentile(samples, 0.95),
                                p99_ms=1000 * helpers.percentile(samples, 0.99),
                                rows_per_s=batch_size / statistics.mean(samples),
                                peak_allocated_extra_bytes=peak_extra,
                                result_sha256=stable[variant],
                                logical_result_tensor_bytes=result_bytes,
                                component_reference_max_abs_error=reference_errors,
                                **helpers.check_memory(torch),
                            )
                        )
                        save()
                tolerance = 2e-6 if stage == "component" else 1e-12
                torch.testing.assert_close(
                    outputs["flat-j4"],
                    outputs["feedback-off-l2"],
                    atol=tolerance,
                    rtol=1e-5 if stage == "component" else 1e-12,
                )
                print(f"{stage} B={batch_size} N={episodes}: comprobado", flush=True)
            if (
                helpers.proof(torch, initial)[0] != initial_proof
                or helpers.proof(torch, batch_payload)[0] != batch_proof
                or helpers.proof(torch, financial_snapshot.export_cpu())[0] != snapshot_proof
                or not torch.equal(rng[0], torch.get_rng_state())
                or not torch.equal(rng[1], torch.cuda.get_rng_state(0))
            ):
                raise AssertionError("La medida alteró entradas, estado, snapshot o RNG")
            for engine in engines.values():
                engine.verify(strong=True)
    if (
        helpers.manifest(root) != pinned
        or digest(HOLD) != HOLD_SHA
        or digest(Path(__file__)) != report["script_sha256"]
        or digest(root / "benchmarks/hierarchical_bridge_cuda.py") != report["public_script_sha256"]
        or digest(root / "benchmarks/regional_decision_cuda.py") != report["helper_sha256"]
    ):
        raise ValueError("Cambiaron las fuentes o la guarda durante la medida")
    report.update(
        status="passed",
        elapsed_s=time.monotonic() - start,
        process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_and_hold_unchanged=True,
    )
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.run:
        if options.output is None:
            raise ValueError("La ejecución requiere una salida privada nueva")
        run(options.repository.resolve(strict=True), options.output.resolve())
    else:
        print(json.dumps(protocol(), indent=2))


if __name__ == "__main__":
    main()
