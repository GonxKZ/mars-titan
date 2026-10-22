"""Preparar o medir el forward completo del puente, sin datos financieros ni ajuste.

Sin --run solo describe el protocolo. La ejecución futura requiere un árbol
verificado, una salida privada nueva y dos procesos supervisados y acotados.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import itertools
import json
import math
import os
import platform
import random
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
SEED, ORDER_SEED = 20261022, 20261023
WARMUP, REPETITIONS = 3, 10
CHILD_SECONDS, TOTAL_SECONDS = 180, 360
ATOL, RTOL = 2e-6, 1e-5
VARIANTS = ("cycles0", "cycles2", "j1", "j2", "j4")
INTERVENTIONS = ("j2", "active-control", "region-removed", "edge-cut", "all-removed")
SHAPES = tuple(itertools.product((1, 256), (3, 8), (8, 1024)))
CODEC, CONTEXT = "a" * 64, "b" * 64
HIDDEN, NEIGHBORS, GAIN = 32, 8, 0.25
ALLOCATOR_CAP, HEADROOM = 512 * 1024**2, 256 * 1024**2


def digest(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def save(path, value):
    if not path.resolve().is_relative_to(BASE):
        raise ValueError("La salida debe permanecer en el área privada del guion")
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


def protocol(suite="nonlinear"):
    variants = INTERVENTIONS if suite == "interventions" else VARIANTS
    return dict(
        kind="shared_evidence_bridge_full_forward_cost_v1"
        if suite == "nonlinear"
        else "shared_evidence_bridge_intervention_forward_cost_v1",
        suite=suite,
        script_sha256=digest(Path(__file__).read_bytes()),
        shapes=[dict(batch=b, regions=r, episodes=n) for b, r, n in SHAPES],
        variants=list(variants),
        conditions_per_process=len(SHAPES) * len(variants),
        passes=["normal", "reverse"],
        warmup=WARMUP,
        repetitions=REPETITIONS,
        seed=SEED,
        order_seed=ORDER_SEED,
        hidden_size=HIDDEN,
        neighbors=NEIGHBORS,
        gain=GAIN,
        gpu_dtype="torch.float32",
        reference_dtype="torch.float64",
        atol=ATOL,
        rtol=RTOL,
        device="cuda:0",
        child_budget_seconds=CHILD_SECONDS,
        total_budget_seconds=TOTAL_SECONDS,
        allocator_cap_bytes=ALLOCATOR_CAP,
        free_memory_headroom_bytes=HEADROOM,
        availability="all_regions_present_no_financial_missingness_claim"
        if suite == "nonlinear"
        else "fixed_masks_and_directed_cuts_dense_work_no_financial_missingness_claim",
        timing_scope="complete_SharedEvidenceBridge_forward_including_candidate_selection",
        percentiles="linear_interpolation_descriptive_only_ten_samples_not_tail_estimates",
        numerical_reference="same_FP32_parameters_inputs_and_labels_lifted_exactly_to_FP64",
        scientific_evaluation=False,
        optimizer=False,
        energy_measured=False,
        physical_launches_measured=False,
        physical_transfers_measured=False,
    )


def manifest(root):
    listed = subprocess.run(
        ["rg", "--files", "src/mars_titan", "-g", "*.py"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    ).stdout.splitlines()
    names = sorted(set(listed + ["pyproject.toml", "uv.lock"]))
    if not names or len(names) > 4096:
        raise ValueError("El árbol fuente no corresponde al límite técnico")
    total, result = 0, {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Una fuente sale del árbol declarado")
        data = path.read_bytes()
        total += len(data)
        if total > 64 * 1024**2:
            raise ValueError("El sello de fuentes supera 64 MiB")
        result[name] = digest(data)
    for name in ("bridge_recurrence", "evidence_bridge"):
        path = root / "src/mars_titan/models/titans" / (name + ".py")
        ast.parse(path.read_bytes(), filename=str(path))
    return result


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Presupuesto agotado antes de iniciar otra operación")


def hold_hash(path):
    with path.open("rb") as stream:
        data = stream.read(64 * 1024 + 1)
    if len(data) > 64 * 1024:
        raise ValueError("La guarda existente supera el límite de lectura técnica")
    return digest(data)


def percentile(values, q):
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summary(samples, batch):
    times = [item["ms"] for item in samples]
    return dict(
        samples=samples,
        p50_ms=statistics.median(times),
        p95_ms=percentile(times, 0.95),
        p99_ms=percentile(times, 0.99),
        mean_ms=statistics.mean(times),
        sample_stdev_ms=statistics.stdev(times),
        rows_per_second=batch * 1000 / statistics.mean(times),
        peak_allocated_bytes=max(item["peak_allocated_bytes"] for item in samples),
        peak_reserved_bytes=max(item["peak_reserved_bytes"] for item in samples),
        peak_allocated_extra_bytes=max(item["allocated_extra_bytes"] for item in samples),
        peak_reserved_extra_bytes=max(item["reserved_extra_bytes"] for item in samples),
    )


def tensor_bytes(value):
    return value.numel() * value.element_size()


def output_bytes(result):
    logical, storages = {}, {}
    for name, value in vars(result).items():
        if hasattr(value, "untyped_storage"):
            logical[name] = tensor_bytes(value)
            storage = value.untyped_storage()
            storages[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return dict(logical=logical, unique_storage_bytes=sum(storages.values()))


def reference_errors(torch, actual, expected):
    result = {}
    for name in ("values", "region_values", "evidence_weights", "bridge_weights"):
        left, right = getattr(actual, name).double(), getattr(expected, name).double()
        difference = (left - right).abs()
        ratio = difference / (ATOL + RTOL * right.abs())
        absolute, relative = float(difference.max()), float(ratio.max())
        result[name] = dict(
            passed=bool(torch.isfinite(left).all() and (ratio <= 1).all()),
            finite_errors=math.isfinite(absolute) and math.isfinite(relative),
            max_absolute_error=absolute if math.isfinite(absolute) else None,
            max_tolerance_ratio=relative if math.isfinite(relative) else None,
        )
    return result


def run_child(root, out, *, reverse, suite="nonlinear"):
    started = time.monotonic()
    deadline = started + CHILD_SECONDS - 8
    report = dict(
        protocol=protocol(suite),
        status="incomplete",
        started_at_utc=now(),
        reverse=reverse,
        cases=[],
        real_corpus=False,
        learning_hold_verification="supervisor_before_after_hashes",
        optimizer_steps=0,
        predictive_improvement_measured=False,
    )
    try:
        pinned = json.loads((out.parent / "source-manifest.json").read_text())
        if manifest(root) != pinned:
            raise ValueError("El árbol fuente cambió antes de la ejecución")
        hardware = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=index,name,uuid,memory.total,memory.used,driver_version",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        concurrent = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        check_budget(deadline)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. No existe sustitución por CPU")
        torch.cuda.set_device("cuda:0")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        if torch.is_autocast_enabled("cuda"):
            raise ValueError("El coste requiere FP32 sin autocast")
        free, total = torch.cuda.mem_get_info("cuda:0")
        allowance = min(ALLOCATOR_CAP, free - HEADROOM)
        if allowance < 128 * 1024**2:
            raise RuntimeError("No queda VRAM suficiente para la reserva técnica")
        torch.cuda.set_per_process_memory_fraction(allowance / total, "cuda:0")
        report["environment"] = dict(
            nvidia_smi=hardware.stdout,
            concurrent_compute_applications=dict(
                code=concurrent.returncode,
                output=concurrent.stdout,
                error=concurrent.stderr,
            ),
            python=sys.version,
            python_executable=sys.executable,
            platform=platform.platform(),
            machine=platform.machine(),
            torch=str(torch.__version__),
            torch_git=torch.version.git_version,
            numpy=importlib.metadata.version("numpy"),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            cpu_threads=torch.get_num_threads(),
            free_vram_before_bytes=free,
            total_vram_bytes=total,
            allocator_limit_bytes=allowance,
            cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            tf32=False,
            autocast=False,
            deterministic_algorithms=True,
        )
        sys.path.insert(0, str(root / "src"))
        from mars_titan.models.titans.bridge_intervention import BridgeInterventionConfig
        from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
        from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot
        from mars_titan.models.titans.evidence_bridge import (
            EvidenceBridgeConfig,
            SharedEvidenceBridge,
        )

        candidate_differences = False
        for case_index, (batch, regions, episodes) in enumerate(SHAPES):
            check_budget(deadline)
            prepared_at = time.perf_counter()
            generator = torch.Generator(device="cpu").manual_seed(SEED + case_index)
            keys = torch.randn((episodes, 64), generator=generator, dtype=torch.float32)
            keys = (keys.double() / keys.double().norm(dim=1, keepdim=True)).float()
            tensors = dict(
                keys=keys,
                values=torch.randn((episodes, 64), generator=generator, dtype=torch.float32),
                labels=torch.randn(episodes, generator=generator, dtype=torch.float32).double(),
                ids=torch.arange(1, episodes + 1, dtype=torch.int64),
                decision_at=torch.full((episodes,), 2, dtype=torch.int64),
                available_at=torch.full((episodes,), 1, dtype=torch.int64),
                maturity_at=torch.full((episodes,), 3, dtype=torch.int64),
            )
            common = dict(codec_id=CODEC, context_id=CONTEXT, cutoff=4)
            source_cpu32 = EpisodeSnapshot.create(**tensors, **common, dtype=torch.float32)
            source = EpisodeSnapshot.restore(
                source_cpu32.export_cpu(),
                **common,
                device="cuda:0",
                dtype=torch.float32,
            )
            source_cpu64 = EpisodeSnapshot.create(**tensors, **common, dtype=torch.float64)
            state = torch.randn((batch, HIDDEN), generator=generator, dtype=torch.float32)
            available = torch.ones((batch, regions), dtype=torch.bool)
            state_gpu, available_gpu = state.to("cuda:0"), available.to("cuda:0")
            options = dict(
                codec_id=CODEC,
                hidden_size=HIDDEN,
                regions=tuple(f"view_{i}" for i in range(regions)),
                neighbors=NEIGHBORS,
                cycles=0,
                bridge_gain=GAIN,
            )
            anchor = SharedEvidenceBridge(EvidenceBridgeConfig(**options), dtype=torch.float32)
            parameters = {name: value.detach().clone() for name, value in anchor.named_parameters()}
            parameter_hashes = {
                name: digest(value.numpy().tobytes()) for name, value in parameters.items()
            }
            del anchor
            entry = dict(
                batch=batch,
                regions=regions,
                episodes=episodes,
                seed=SEED + case_index,
                source_payload_sha256={
                    name: digest(value.numpy().tobytes()) for name, value in tensors.items()
                },
                source_cpu32_sha256=source_cpu32._digest,
                source_cuda32_sha256=source._digest,
                source_cpu64_sha256=source_cpu64._digest,
                parameter_sha256=parameter_hashes,
                variants={},
                orders=[],
                snapshot_bytes=source.usage(),
            )
            report["cases"].append(entry)
            models, first_ids = {}, None
            variants = INTERVENTIONS if suite == "interventions" else VARIANTS
            for variant in variants:
                check_budget(deadline)
                intervention = None
                if suite == "interventions" and variant != "j2":
                    active = tuple(
                        variant != "all-removed" and not (variant == "region-removed" and i == 1)
                        for i in range(regions)
                    )
                    intervention = BridgeInterventionConfig(
                        active, ((0, 1),) if variant == "edge-cut" else ()
                    )
                config = EvidenceBridgeConfig(
                    **dict(options, cycles=2 if variant == "cycles2" else 0),
                    recurrence=None
                    if variant.startswith("cycles")
                    else BridgeRecurrenceConfig(
                        2 if suite == "interventions" else int(variant[1:]), 512
                    ),
                    intervention=intervention,
                )
                model = SharedEvidenceBridge(config, dtype=torch.float32)
                cpu = SharedEvidenceBridge(config, dtype=torch.float64)
                with torch.no_grad():
                    for name, value in model.named_parameters():
                        value.copy_(parameters[name])
                    for name, value in cpu.named_parameters():
                        value.copy_(parameters[name].double())
                model.eval().requires_grad_(False)
                cpu.eval().requires_grad_(False)
                models[variant] = model.to("cuda:0")
                with torch.no_grad():
                    expected = cpu(
                        state.double(),
                        source_cpu64,
                        **dict(context_id=CONTEXT, cutoff=4),
                        available=available,
                    )
                    actual = model(
                        state_gpu,
                        source,
                        context_id=CONTEXT,
                        cutoff=4,
                        available=available_gpu,
                    )
                    read_bytes = output_bytes(actual)
                    actual_cpu = type(actual)(
                        **{
                            name: value.cpu() if isinstance(value, torch.Tensor) else value
                            for name, value in vars(actual).items()
                        }
                    )
                    ids = actual_cpu.evidence_ids
                    if first_ids is not None and not torch.equal(first_ids, ids):
                        raise ValueError(
                            "Dos variantes emparejadas seleccionan candidatos diferentes"
                        )
                    first_ids = ids.clone()
                    same_ids = torch.equal(ids, expected.evidence_ids)
                    differing = (ids != expected.evidence_ids).any(dim=1)
                    difference_rows = int(differing.sum())
                    examples = [
                        dict(
                            row=index,
                            cuda32=ids[index].tolist(),
                            cpu64=expected.evidence_ids[index].tolist(),
                        )
                        for index in differing.nonzero().flatten().tolist()[:4]
                    ]
                    full_errors = reference_errors(torch, actual_cpu, expected)
                    conditioned = None
                    if not same_ids:
                        candidate_differences = True
                        conditioned = reference_errors(
                            torch,
                            actual_cpu,
                            cpu(
                                state.double(),
                                source_cpu64,
                                context_id=CONTEXT,
                                cutoff=4,
                                available=available,
                                positions=actual_cpu.selected_positions,
                            ),
                        )
                    checked = full_errors if same_ids else conditioned
                    passed = all(value["passed"] for value in checked.values())
                    entry["variants"][variant] = dict(
                        configuration=config.identity(),
                        full_reference_passed=same_ids and passed,
                        same_candidates=same_ids,
                        different_candidate_rows=difference_rows,
                        candidate_difference_examples=examples,
                        cuda_candidate_ids_sha256=digest(ids.contiguous().numpy().tobytes()),
                        cpu64_candidate_ids_sha256=digest(
                            expected.evidence_ids.contiguous().numpy().tobytes()
                        ),
                        full_reference_errors=full_errors,
                        errors_with_gpu_candidates=conditioned,
                        numerical_parity_with_common_candidates=passed,
                        output_bytes=read_bytes,
                        parameter_bytes=sum(tensor_bytes(value) for value in model.parameters()),
                        buffer_bytes=sum(tensor_bytes(value) for value in model.buffers()),
                        estimated_working_bytes=model.estimated_bytes(batch, episodes),
                        warmup_samples=[],
                        measured_samples=[],
                    )
                    if not passed:
                        raise ArithmeticError("El forward no cumple la tolerancia fijada")
                del cpu, expected, actual, actual_cpu
            torch.cuda.synchronize("cuda:0")
            entry["preparation_and_cpu_reference_seconds"] = time.perf_counter() - prepared_at
            sizes = source.usage()
            entry["accounted_initial_h2d_bytes"] = (
                sum(sizes[name] for name in ("keys", "values", "labels", "ids"))
                + tensor_bytes(state)
                + tensor_bytes(available)
                + sum(
                    item["parameter_bytes"] + item["buffer_bytes"]
                    for item in entry["variants"].values()
                )
            )
            order = list(variants)
            random.Random(ORDER_SEED + case_index).shuffle(order)
            if reverse:
                order.reverse()
            for iteration in range(WARMUP + REPETITIONS):
                shifted = order[iteration % len(order) :] + order[: iteration % len(order)]
                if iteration % 2:
                    shifted.reverse()
                entry["orders"].append(shifted)
                for variant in shifted:
                    check_budget(deadline)
                    torch.cuda.synchronize("cuda:0")
                    allocated, reserved = (
                        torch.cuda.memory_allocated(0),
                        torch.cuda.memory_reserved(0),
                    )
                    torch.cuda.reset_peak_memory_stats(0)
                    tick = time.perf_counter_ns()
                    with torch.no_grad():
                        result = models[variant](
                            state_gpu,
                            source,
                            context_id=CONTEXT,
                            cutoff=4,
                            available=available_gpu,
                        )
                    torch.cuda.synchronize("cuda:0")
                    elapsed = (time.perf_counter_ns() - tick) / 1e6
                    peak_allocated, peak_reserved = (
                        torch.cuda.max_memory_allocated(0),
                        torch.cuda.max_memory_reserved(0),
                    )
                    item = dict(
                        ms=elapsed,
                        allocated_before_bytes=allocated,
                        reserved_before_bytes=reserved,
                        peak_allocated_bytes=peak_allocated,
                        peak_reserved_bytes=peak_reserved,
                        allocated_extra_bytes=max(0, peak_allocated - allocated),
                        reserved_extra_bytes=max(0, peak_reserved - reserved),
                    )
                    bucket = "warmup_samples" if iteration < WARMUP else "measured_samples"
                    entry["variants"][variant][bucket].append(item)
                    del result
            for item in entry["variants"].values():
                item["measurement"] = summary(item.pop("measured_samples"), batch)
            entry["resident_gpu_parameter_copies"] = len(models)
            source.verify()
            source_cpu32.verify()
            source_cpu64.verify()
            save(out / "results.json", report)
            print(
                json.dumps(
                    dict(
                        case=case_index,
                        batch=batch,
                        regions=regions,
                        episodes=episodes,
                        completed=True,
                    )
                ),
                flush=True,
            )
            del (
                models,
                model,
                source,
                source_cpu32,
                source_cpu64,
                parameters,
                tensors,
                state_gpu,
                available_gpu,
            )
            torch.cuda.empty_cache()
        if manifest(root) != pinned:
            raise ValueError("El árbol fuente cambió durante la medición")
        if digest(Path(__file__).read_bytes()) != report["protocol"]["script_sha256"]:
            raise ValueError("El guion cambió durante la medición")
        report["status"] = "completed_candidate_differences" if candidate_differences else "passed"
    except BaseException as error:
        report["error"] = dict(type=type(error).__name__, reason=str(error))
        traceback.print_exc()
    finally:
        report.update(
            finished_at_utc=now(),
            process_seconds=time.monotonic() - started,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
        )
        save(out / "results.json", report)
    return 0 if report["status"] == "passed" else 1


def supervise(root, output, *, suite="nonlinear"):
    started = time.monotonic()
    root, output = root.resolve(), output.resolve()
    if not output.is_relative_to(BASE) or output.is_relative_to(root):
        raise ValueError("La ejecución necesita una salida nueva en el área privada del guion")
    if output.exists() or not output.parent.is_dir():
        raise ValueError("La salida debe ser nueva y su carpeta superior debe existir")
    pinned = manifest(root)
    hold = (
        Path(
            os.environ.get(
                "MARS_TITAN_TRAINING_HOLD",
                Path.home() / ".local/state/mars-titan/training-hold-2000.json",
            )
        )
        .expanduser()
        .resolve(strict=True)
    )
    before = hold_hash(hold)
    output.mkdir(mode=0o700)
    save(output / "source-manifest.json", pinned)
    execution = dict(
        protocol=protocol(suite),
        source_root=str(root),
        started_at_utc=now(),
        status="incomplete",
        learning_hold_path=str(hold),
        learning_hold_sha256_before=before,
        children=[],
    )
    try:
        for reverse in (False, True):
            if hold_hash(hold) != before:
                raise ValueError("La guarda global cambió antes de iniciar otro hijo")
            remaining = TOTAL_SECONDS - (time.monotonic() - started)
            if remaining <= 8:
                raise TimeoutError("No se inicia otro hijo sin presupuesto global")
            out = output / ("reverse" if reverse else "normal")
            out.mkdir(mode=0o700)
            command = [
                "uv",
                "run",
                "--no-project",
                "--no-config",
                "--no-env-file",
                "--offline",
                "--no-python-downloads",
                "--python",
                sys.executable,
                sys.executable,
                str(Path(__file__).resolve()),
                "--source-root",
                str(root),
                "--child",
                str(out),
                "--suite",
                suite,
            ]
            if reverse:
                command.append("--reverse")
            environment = dict(
                os.environ,
                CUDA_VISIBLE_DEVICES="0",
                CUBLAS_WORKSPACE_CONFIG=":4096:8",
                OMP_NUM_THREADS="1",
                MKL_NUM_THREADS="1",
                PYTHONDONTWRITEBYTECODE="1",
                UV_NO_CACHE="1",
                MARS_TITAN_TRAINING_HOLD=str(hold),
            )
            child_started = time.monotonic()
            process, code, timed_out = None, None, False
            try:
                with (out / "execution.log").open("x", encoding="utf-8") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=out,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    try:
                        code = process.wait(timeout=min(CHILD_SECONDS, remaining - 3))
                    except subprocess.TimeoutExpired:
                        timed_out = True
            finally:
                if process is not None and code is None:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    code = process.wait(timeout=3)
                result = out / "results.json"
                execution["children"].append(
                    dict(
                        reverse=reverse,
                        command=command,
                        exit_code=code,
                        timed_out=timed_out,
                        external_process_seconds=time.monotonic() - child_started,
                        results_sha256=digest(result.read_bytes()) if result.is_file() else None,
                    )
                )
                save(output / "execution.json", execution)
            if code != 0 or timed_out:
                raise RuntimeError("Un hijo no completó la paridad y las condiciones previstas")
            if hold_hash(hold) != before:
                raise ValueError("La guarda global cambió durante la medición")
        execution["status"] = "passed"
    except BaseException as error:
        execution["error"] = dict(type=type(error).__name__, reason=str(error))
    finally:
        try:
            after = hold_hash(hold)
            execution["learning_hold_sha256_after"] = after
            execution["learning_hold_before_after_equal"] = after == before
            if after != before:
                execution["status"] = "incomplete_hold_changed"
        except (OSError, ValueError) as error:
            execution["learning_hold_after_error"] = str(error)
            execution["status"] = "incomplete_hold_unreadable"
        execution.update(finished_at_utc=now(), external_total_seconds=time.monotonic() - started)
        save(output / "execution.json", execution)
    print(json.dumps(dict(status=execution["status"], output=str(output))), flush=True)
    return 0 if execution["status"] == "passed" else 1


def terminate(signum, _frame):
    raise SystemExit(128 + signum)


def main():
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--suite", choices=("nonlinear", "interventions"), default="nonlinear")
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--reverse", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child is not None:
        if args.source_root is None or args.run or args.output is not None:
            parser.error("El hijo necesita únicamente árbol, salida interna y orden")
        return run_child(
            args.source_root.resolve(), args.child.resolve(), reverse=args.reverse, suite=args.suite
        )
    if args.run:
        if args.source_root is None or args.output is None:
            parser.error("--run necesita --source-root y --output")
        return supervise(args.source_root, args.output, suite=args.suite)
    if args.output is not None or args.reverse:
        parser.error("Describir el protocolo no escribe una salida ni ejecuta variantes")
    print(json.dumps(protocol(args.suite), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
