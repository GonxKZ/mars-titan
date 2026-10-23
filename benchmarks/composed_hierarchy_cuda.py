"""Sonda exploratoria del consumidor residente, sin aprendizaje ni sesión durable.

Sin --run solo imprime el protocolo mediante stdlib. Las tres muestras por orden
no estiman colas con precisión fiable. Cada control tiene pesos y estado propios.
La ejecución exige una copia privada, --repository y --output externo nuevo.
"""

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import platform
import signal
import statistics
import sys
import time
from collections.abc import Mapping
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

BASE = Path(__file__).resolve().parent
ARMS = ("flat", "pre_only", "post_only", "both")
SIZES = (4, 64, 256)
WARMUP, REPEATS, BLOCK, INTERNAL_SECONDS = 1, 3, 64, 480


def protocol():
    return dict(
        kind="joint_hierarchy_resident_cost_v1",
        controls=list(ARMS),
        rows=list(SIZES),
        device="cuda:0",
        dtype="torch.float64",
        hidden_size=32,
        seed=42,
        fixture_seed=23,
        fixture_event=0,
        cpu_threads=2,
        interop_threads=1,
        predictor_max_batch=256,
        reader_max_batch=64,
        physical_block_rows=BLOCK,
        pre="RegionalHierarchyConfig(tree), self/uniform operator, two rounds",
        post="BridgeHierarchyConfig(tree,local_steps=2), four views, quota988",
        flat="four views, BridgeRecurrenceConfig(steps=4), quota988, cycles0",
        action="off/k1/none",
        pt4=False,
        episodes=8,
        optimizer_steps=0,
        orders=[list(ARMS), list(reversed(ARMS))],
        warmup=WARMUP,
        samples=REPEATS,
        records=24,
        cohort_invocations=96,
        physical_prepare_invocations=192,
        timing="begin + all physical prepare blocks + finish, CUDA synchronized",
        percentiles="linear interpolation of three exploratory samples per order",
        tail_precision_claim=False,
        predictive_comparison=False,
        paired_parameters=False,
        allocator_cap_bytes=512 * 1024**2,
        free_margin_bytes=256 * 1024**2,
        retained_cpu_tensor_cap_bytes=64 * 1024**2,
        internal_seconds=INTERNAL_SECONDS,
        external_seconds=510,
        initial_state="new fast state, no own prefix, common mature fixture snapshot",
        excluded=[
            "durable commit/reopen/prune",
            "collection",
            "feedback",
            "corpus encoding",
            "initial construction from sample latency",
            "energy",
            "physical transfers",
        ],
    )


def load_helper(root):
    spec = importlib.util.spec_from_file_location(
        "joint_cost_helpers", root / "benchmarks/regional_decision_cuda.py"
    )
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper


def plain(torch, np, value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return dict(array_dtype=str(value.dtype), shape=list(value.shape), values=value.tolist())
    if is_dataclass(value):
        return dict(
            type=type(value).__module__ + "." + type(value).__qualname__,
            fields={
                field.name: plain(torch, np, getattr(value, field.name)) for field in fields(value)
            },
        )
    if isinstance(value, Mapping):
        return {key: plain(torch, np, item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(plain(torch, np, item) for item in value)
    return value


def construct(torch, arm):
    from test_bridge_episodic_readout import configuration
    from test_financial_adapter import specification

    from mars_titan.models.titans.bridge_hierarchy import BridgeHierarchyConfig
    from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
    from mars_titan.models.titans.episodic_readout import EpisodicReadout
    from mars_titan.models.titans.financial import FinancialConfig, FinancialPredictor
    from mars_titan.models.titans.mechanism_catalog import MechanismAction, MechanismCatalog
    from mars_titan.models.titans.mechanism_consumer import FrozenMechanismConsumer
    from mars_titan.models.titans.regional_fusion import RegionalFusionConfig
    from mars_titan.models.titans.regional_hierarchy import RegionalHierarchyConfig

    pre, post = arm in ("pre_only", "both"), arm in ("post_only", "both")
    config = FinancialConfig(
        specification(),
        variant="mac_online",
        hidden_size=32,
        seed=42,
        max_batch=256,
        head="quantile_head_v1",
        regional_fusion=RegionalFusionConfig(topology="self", routing="uniform") if pre else None,
        regional_execution=RegionalHierarchyConfig() if pre else None,
    )
    model = (
        FinancialPredictor(config, dtype=torch.float64).to("cuda:0").eval().requires_grad_(False)
    )
    base = configuration()
    reader_config = replace(
        base,
        max_batch=64,
        evidence_bridge=replace(
            base.evidence_bridge,
            max_batch=64,
            regions=("v0", "v1", "v2", "v3"),
            topology="self",
            cycles=0,
            recurrence=None if post else BridgeRecurrenceConfig(4, 988),
            hierarchy=BridgeHierarchyConfig(max_node_updates_per_row=988) if post else None,
        ),
    )
    reader = (
        EpisodicReadout(reader_config, device="cuda:0", dtype=torch.float64)
        .eval()
        .requires_grad_(False)
    )
    catalog = MechanismCatalog(
        reader_contract=reader_config.evidence_bridge.identity()["kind"],
        regional_execution=config.regional_execution,
    )
    engine = FrozenMechanismConsumer(
        model, readout=reader, action=MechanismAction(), catalog=catalog
    )
    if pre and post and engine.identity()["schema_version"] != 6:
        raise AssertionError("La composición no conserva su identidad explícita de esquema6")
    return engine


def cohort(engine, blocks, states, auxiliary, snapshot, context_id):
    flows = tuple(flow for batch in blocks for flow in batch.flow_ids)
    cutoff = blocks[0].prediction_at[0]
    event = engine.begin(
        auxiliary,
        flow_ids=flows,
        cutoff=cutoff,
        context_id=context_id,
        matures_at=tuple(cutoff + 86400000000 for _ in flows),
        snapshot=snapshot,
    )
    following = []
    for batch, state in zip(blocks, states, strict=True):
        result = engine.prepare(batch, state, event)
        event = result.event
        following.append(result.next_state)
    return engine.finish(event), tuple(following)


def run(root, output):
    root, output = root.resolve(strict=True), output.resolve()
    if not root.is_dir():
        raise ValueError("El repositorio debe ser un directorio existente")
    if Path(__file__).resolve().is_relative_to(root):
        raise ValueError("Ejecutar una copia privada del guion, fuera del repositorio")
    if output.exists() or not output.parent.is_dir() or output.is_relative_to(root):
        raise ValueError("La salida debe ser nueva, con padre existente y fuera del repositorio")
    canonical = root / "benchmarks/composed_hierarchy_cuda.py"
    if not canonical.is_file():
        raise ValueError("Falta la fuente pública benchmarks/composed_hierarchy_cuda.py")
    if (
        hashlib.sha256(Path(__file__).read_bytes()).digest()
        != hashlib.sha256(canonical.read_bytes()).digest()
    ):
        raise ValueError("La copia difiere de su fuente pública")
    started = time.monotonic()
    helper = load_helper(root)
    pinned = helper.manifest(root)
    hold = helper.check_hold()
    for path in (root, root / "src", root / "tests/models/titans"):
        sys.path.insert(0, str(path))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import numpy as np
    import torch
    from test_context_selection import batch_on
    from test_episodic_readout import CODEC, CONTEXT, source

    from mars_titan.models.titans import financial
    from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
    from mars_titan.training.learning_hold import install_optimizer_guard

    if not Path(financial.__file__).resolve().is_relative_to(root / "src"):
        raise RuntimeError("La sonda importó otro árbol")
    hardware = helper.nvidia()
    if not torch.cuda.is_available():
        raise RuntimeError("Falta CUDA, sin sustitución por CPU")
    torch.cuda.set_device("cuda:0")
    if "4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("cuda:0 no es la RTX4070 prevista")
    free, total = torch.cuda.mem_get_info(0)
    if free < 768 * 1024**2:
        raise MemoryError("Falta reserva inicial de VRAM")
    torch.cuda.set_per_process_memory_fraction(512 * 1024**2 / total, 0)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    attempts = []

    def blocked(message):
        attempts.append(message)
        raise AssertionError(message)

    guard = install_optimizer_guard(blocked, path=helper.HOLD)
    if guard is None:
        raise RuntimeError("La guarda de optimizador no quedó instalada")
    report = dict(
        protocol=protocol(),
        status="incomplete",
        records=[],
        source_sha256=pinned,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=hashlib.sha256(
            (root / "benchmarks/regional_decision_cuda.py").read_bytes()
        ).hexdigest(),
        hold_sha256_start=hold,
        python=platform.python_version(),
        torch=str(torch.__version__),
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(0),
        hardware_before=hardware,
        cpu_count=os.cpu_count(),
        loadavg_start=os.getloadavg(),
        ram_start=helper.ram(),
    )

    def save():
        temporary = output.with_name(output.name + ".writing")
        with temporary.open("x") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(output)

    def deadline(*unused):
        raise TimeoutError("La sonda agotó 480 s internos")

    previous_handler = signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(
        signal.ITIMER_REAL, max(0.001, INTERNAL_SECONDS - (time.monotonic() - started))
    )
    parameter_baselines, references = {}, {}
    try:
        save()
        with torch.no_grad():
            for size in SIZES:
                full_batch = None
                for order_index, order in enumerate((ARMS, tuple(reversed(ARMS)))):
                    for arm in order:
                        init_started = time.monotonic()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats(0)
                        engine = construct(torch, arm)
                        if full_batch is None:
                            full_batch = batch_on(
                                engine.predictor, tuple(f"US/F{i:04d}" for i in range(size)), 0
                            )
                        blocks = tuple(
                            full_batch.select(list(range(i, min(i + BLOCK, size))))
                            for i in range(0, size, BLOCK)
                        )
                        states = tuple(
                            engine.predictor.initial_state(batch.flow_ids) for batch in blocks
                        )
                        auxiliary = engine.initial_state()
                        snapshot = EpisodeSnapshot.create(
                            **source(8),
                            codec_id=CODEC,
                            context_id=CONTEXT,
                            cutoff=full_batch.prediction_at[0],
                            device="cuda:0",
                            dtype=torch.float64,
                        )
                        parameters = {
                            prefix + name: dict(
                                shape=list(value.shape),
                                numel=value.numel(),
                                dtype=str(value.dtype),
                                sha256=helper.proof(torch, value)[0],
                            )
                            for prefix, module in (
                                ("predictor/", engine.predictor),
                                ("reader/", engine.readout),
                            )
                            for name, value in module.named_parameters()
                        }
                        parameter_baselines.setdefault(size, parameters)
                        common = parameter_baselines[size].keys() & parameters.keys()
                        frozen = dict(
                            model=engine.predictor.state_dict(),
                            reader=engine.readout.state_dict(),
                            inputs={
                                f.name: getattr(full_batch, f.name)
                                for f in fields(full_batch)
                                if f.name != "_versions"
                            },
                            blocks=tuple(
                                {
                                    f.name: getattr(batch, f.name)
                                    for f in fields(batch)
                                    if f.name != "_versions"
                                }
                                for batch in blocks
                            ),
                            states=states,
                            auxiliary=auxiliary,
                            snapshot=snapshot.export_cpu(),
                        )
                        frozen_sha, frozen_bytes, _ = helper.proof(torch, plain(torch, np, frozen))
                        rng_cpu, rng_gpu = (
                            torch.random.get_rng_state().clone(),
                            torch.cuda.get_rng_state(0).clone(),
                        )
                        record = dict(
                            arm=arm,
                            rows=size,
                            order=order_index,
                            actual_order=list(order),
                            status="incomplete",
                            init_seconds=time.monotonic() - init_started,
                            construction_memory=helper.check_memory(torch),
                            parameter_count=sum(v["numel"] for v in parameters.values()),
                            parameters=parameters,
                            state_logical_tensor_bytes=helper.proof(torch, (states, auxiliary))[1],
                            shared_names=len(common),
                            changed_common_names=sorted(
                                name
                                for name in common
                                if parameters[name] != parameter_baselines[size][name]
                            ),
                            shared_equal=sum(
                                parameters[n] == parameter_baselines[size][n] for n in common
                            ),
                            extra_names=sorted(
                                parameters.keys() - parameter_baselines[size].keys()
                            ),
                            missing_names=sorted(
                                parameter_baselines[size].keys() - parameters.keys()
                            ),
                            frozen_logical_tensor_bytes=frozen_bytes,
                            frozen_sha256=frozen_sha,
                            consumer_id=engine.model_id,
                            buffers={
                                prefix + name: dict(
                                    shape=list(v.shape),
                                    dtype=str(v.dtype),
                                    bytes=v.numel() * v.element_size(),
                                )
                                for prefix, module in (
                                    ("predictor/", engine.predictor),
                                    ("reader/", engine.readout),
                                )
                                for name, v in module.named_buffers()
                            },
                            ram_before=helper.ram(),
                            warmup=[],
                            samples=[],
                        )
                        report["records"].append(record)
                        reference = None
                        for iteration in range(WARMUP + REPEATS):
                            helper.check_memory(torch)
                            torch.cuda.synchronize(0)
                            torch.cuda.reset_peak_memory_stats(0)
                            tick = time.perf_counter()
                            result = cohort(engine, blocks, states, auxiliary, snapshot, CONTEXT)
                            torch.cuda.synchronize(0)
                            elapsed = time.perf_counter() - tick
                            memory = helper.check_memory(torch)
                            work = dict(result[0].work)
                            expected = dict(
                                core_observations=size,
                                mac_online_observations=size,
                                reader_refinements=size,
                                bridge_initial_views=4 * size,
                                bridge_node_updates=(26 if arm in ("post_only", "both") else 16)
                                * size,
                                regional_local_node_evaluations=20 * size
                                if arm in ("pre_only", "both")
                                else 0,
                                regional_summary_evaluations=10 * size
                                if arm in ("pre_only", "both")
                                else 0,
                            )
                            if any(work.get(name, 0) != value for name, value in expected.items()):
                                raise AssertionError(
                                    "El trabajo observado no corresponde a la configuración"
                                )
                            if any(
                                not torch.equal(
                                    state.observed_steps, torch.ones_like(state.observed_steps)
                                )
                                for state in result[1]
                            ):
                                raise AssertionError("El núcleo no avanzó una vez por observación")
                            result_sha, logical_bytes, retained = helper.proof(
                                torch, plain(torch, np, result), keep=True
                            )
                            if logical_bytes > 64 * 1024**2:
                                raise MemoryError(
                                    "La referencia CPU supera64MiB lógicos de tensores"
                                )
                            if reference is None:
                                reference = retained
                            else:
                                helper.strict_equal(torch, reference, retained)
                            key = (size, arm)
                            references.setdefault(key, (result_sha, frozen_sha))
                            if references[key] != (result_sha, frozen_sha):
                                raise AssertionError("El orden cambió resultado o estado inicial")
                            if not torch.equal(
                                rng_cpu, torch.random.get_rng_state()
                            ) or not torch.equal(rng_gpu, torch.cuda.get_rng_state(0)):
                                raise AssertionError("La preparación cambió el RNG")
                            item = dict(
                                seconds=elapsed,
                                rows_per_second=size / elapsed,
                                memory=memory,
                                ram=helper.ram(),
                            )
                            record["warmup" if iteration < WARMUP else "samples"].append(item)
                            record.update(
                                result_sha256=result_sha,
                                result_logical_tensor_bytes=logical_bytes,
                                work=dict(result[0].work),
                            )
                            del result, retained
                        if helper.proof(torch, plain(torch, np, frozen))[0] != frozen_sha:
                            raise AssertionError(
                                "La preparación alteró parámetros, inputs o estados"
                            )
                        samples = [item["seconds"] for item in record["samples"]]
                        record.update(
                            status="passed",
                            min_seconds=min(samples),
                            median_seconds=statistics.median(samples),
                            max_seconds=max(samples),
                            p50_seconds=helper.percentile(samples, 0.50),
                            p95_seconds=helper.percentile(samples, 0.95),
                            p99_seconds=helper.percentile(samples, 0.99),
                            throughput_rows_per_second=size / statistics.median(samples),
                            ram_after=helper.ram(),
                        )
                        save()
                        del reference, frozen, states, blocks, snapshot, engine
                        gc.collect()
                del full_batch
        if len(report["records"]) != 24 or attempts:
            raise AssertionError("La sonda no completó el protocolo sin optimizador")
        report["status"] = "passed"
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        guard.remove()
        report.update(
            loadavg_end=os.getloadavg(),
            ram_end=helper.ram(),
            optimizer_attempts=len(attempts),
            hold_sha256_end=helper.check_hold(),
        )
        report["sources_unchanged"] = helper.manifest(root) == pinned
        report["hardware_after"] = helper.nvidia()
        report["elapsed_script_seconds"] = time.monotonic() - started
        report["script_unchanged"] = (
            hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == report["script_sha256"]
        )
        report["helper_unchanged"] = (
            hashlib.sha256((root / "benchmarks/regional_decision_cuda.py").read_bytes()).hexdigest()
            == report["helper_sha256"]
        )
        if (
            report["elapsed_script_seconds"] > INTERNAL_SECONDS
            or not report["sources_unchanged"]
            or not report["script_unchanged"]
            or not report["helper_unchanged"]
        ):
            report["status"] = "incomplete"
        save()
    if report["status"] != "passed":
        raise RuntimeError("La sonda quedó incompleta")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repository", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.run:
        print(json.dumps(protocol(), indent=2))
        return
    if args.repository is None or args.output is None:
        parser.error("--run requiere --repository y --output")
    run(args.repository, args.output)


if __name__ == "__main__":
    main()
