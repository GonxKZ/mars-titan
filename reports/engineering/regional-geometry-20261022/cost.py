"""Medir la carga y preparación de fixtures, con ejecución explícita."""

import argparse
import gc
import hashlib
import json
import math
import resource
import statistics
import subprocess
import sys
import time
from dataclasses import fields, is_dataclass
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
PROTOCOL = BASE / "cost-protocol.json"
HOLD = Path.home() / ".local/state/mars-titan/training-hold-2000.json"
sys.dont_write_bytecode = True


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    temporary = path.with_suffix(".temporary")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sources(repository, archive=None):
    names = subprocess.run(
        ["rg", "--files", "src", "tests", "configs", "pyproject.toml", "uv.lock"],
        cwd=repository,
        capture_output=True,
        text=True,
        check=True,
        timeout=8,
    ).stdout.splitlines()
    if not names or len(names) > 4096:
        raise ValueError("El conjunto de fuentes no está acotado")
    result, total = {}, 0
    for name in sorted(set(names)):
        path = repository / name
        if path.is_symlink() or not path.resolve().is_relative_to(repository):
            raise ValueError("Una fuente sale del repositorio declarado")
        data = path.read_bytes()
        total += len(data)
        if total > 128 * 1024**2:
            raise ValueError("Las fuentes superan 128 MiB")
        result[name] = dict(sha256=sha(data), bytes=len(data))
        if archive is not None:
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    return result


def check_hold(protocol):
    data = HOLD.read_bytes()
    if len(data) > 65536 or sha(data) != protocol["hold_sha256"]:
        raise ValueError("La guarda global cambió")
    if json.loads(data).get("training_allowed") is not False:
        raise ValueError("La guarda no conserva el aprendizaje cerrado")
    return sha(data)


def rss():
    rows = Path("/proc/self/status").read_text().splitlines()
    current = next(int(row.split()[1]) * 1024 for row in rows if row.startswith("VmRSS:"))
    return dict(
        current_bytes=current,
        process_lifetime_peak_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )


def fingerprint(torch, value):
    """Incluir tensores, cursores y metadatos fuera del intervalo medido."""
    digest, tensor_bytes = hashlib.sha256(), 0

    def visit(part):
        nonlocal tensor_bytes
        if isinstance(part, torch.Tensor):
            if part.requires_grad or part.grad_fn is not None or not torch.isfinite(part).all():
                raise AssertionError("Un tensor contiene un grafo o un valor no finito")
            cpu = part.detach().cpu().contiguous()
            digest.update(json.dumps(["tensor", list(cpu.shape), str(cpu.dtype)]).encode())
            digest.update(cpu.numpy().tobytes())
            tensor_bytes += cpu.numel() * cpu.element_size()
        elif is_dataclass(part):
            digest.update(type(part).__qualname__.encode())
            visit({field.name: getattr(part, field.name) for field in fields(part)})
        elif isinstance(part, dict):
            digest.update(b"dict")
            for key in sorted(part):
                visit(key)
                visit(part[key])
        elif isinstance(part, (tuple, list)):
            digest.update(json.dumps([type(part).__name__, len(part)]).encode())
            for item in part:
                visit(item)
        elif part is None or type(part) in (str, int, float, bool):
            digest.update(json.dumps([type(part).__name__, part], allow_nan=False).encode())
        else:
            raise TypeError(f"Campo no previsto: {type(part).__name__}")

    visit(value)
    return dict(sha256=digest.hexdigest(), logical_tensor_bytes=tensor_bytes)


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def memory(torch, protocol):
    free, total = torch.cuda.mem_get_info(0)
    result = dict(
        allocated_bytes=torch.cuda.memory_allocated(0),
        reserved_bytes=torch.cuda.memory_reserved(0),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
        global_free_bytes=free,
        global_total_bytes=total,
    )
    if (
        max(result[key] for key in result if "allocated" in key or "reserved" in key)
        > protocol["torch_allocator_cap_bytes"]
    ):
        raise MemoryError("El asignador de Torch superó la cuota")
    if free < protocol["minimum_global_free_vram_bytes"]:
        raise MemoryError("La reserva global de VRAM es insuficiente")
    return result


def measure(torch, operation, protocol, deadline, verify):
    samples, expected = [], None
    for index in range(protocol["warmup_per_record"] + protocol["measured_repetitions_per_record"]):
        if time.monotonic() >= deadline:
            raise TimeoutError("El benchmark alcanzó su límite interno")
        torch.cuda.synchronize(0)
        memory(torch, protocol)
        torch.cuda.reset_peak_memory_stats(0)
        before, ram_before = memory(torch, protocol), rss()
        started = time.perf_counter_ns()
        result = operation()
        torch.cuda.synchronize(0)
        elapsed = (time.perf_counter_ns() - started) / 1e9
        measured, ram_after = memory(torch, protocol), rss()
        proof = verify(result)
        if expected is not None and proof != expected:
            raise AssertionError("Las repeticiones no conservaron el resultado exacto")
        expected = proof
        if index >= protocol["warmup_per_record"]:
            samples.append(
                dict(
                    seconds=elapsed,
                    torch_allocator=measured,
                    peak_allocated_extra_bytes=measured["peak_allocated_bytes"]
                    - before["allocated_bytes"],
                    peak_reserved_extra_bytes=measured["peak_reserved_bytes"]
                    - before["reserved_bytes"],
                    rss_before=ram_before,
                    rss_after=ram_after,
                )
            )
        del result
    seconds = [sample["seconds"] for sample in samples]
    return dict(
        warmup_completed=protocol["warmup_per_record"],
        samples=samples,
        result=expected,
        p50_ms=statistics.median(seconds) * 1000,
        p95_ms=percentile(seconds, 0.95) * 1000,
        p99_ms=percentile(seconds, 0.99) * 1000,
        mean_ms=statistics.mean(seconds) * 1000,
        min_ms=min(seconds) * 1000,
        max_ms=max(seconds) * 1000,
        sample_stdev_ms=statistics.stdev(seconds) * 1000,
        timed_total_seconds=sum(seconds),
        calls_per_second=1 / statistics.mean(seconds),
    )


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--label", choices=("before", "after"), required=True)
    args = parser.parse_args()
    repository = args.repository.resolve(strict=True)
    output = BASE / f"cost-{args.label}"
    output.mkdir(exist_ok=False)
    protocol_bytes = PROTOCOL.read_bytes()
    protocol = json.loads(protocol_bytes)
    deadline = started + protocol["internal_deadline_seconds"]
    report = dict(
        status="incomplete",
        label=args.label,
        repository=str(repository),
        started_at_utc=datetime.now(UTC).isoformat(),
        records=[],
        protocol_sha256=sha(protocol_bytes),
        script_sha256=sha(Path(__file__).read_bytes()),
    )
    try:
        check_hold(protocol)
        manifest = sources(repository, output / "sources")
        if args.label == "before":
            baseline = json.loads((BASE / "before-workspace-manifest.json").read_text())
            if manifest != baseline["files"]:
                raise ValueError("La referencia anterior no coincide con su copia fijada")
        save(output / "source-manifest.json", manifest)
        (output / "protocol.json").write_bytes(protocol_bytes)
        (output / "cost.py").write_bytes(Path(__file__).read_bytes())
        save(output / "results.json", report)
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=name,driver_version,memory.total,memory.free",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        applications = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        sys.path[:0] = [
            str(repository / "src"),
            str(repository),
            str(repository / "tests/models/titans"),
            str(repository / "tests/memory"),
        ]
        import test_context_selection
        import test_market_memory
        import torch

        from mars_titan.models.titans.financial import FinancialPredictor
        from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer
        from mars_titan.models.titans.regional_fusion import RegionalFusionConfig

        for name, module in tuple(sys.modules.items()):
            if (
                name == "mars_titan"
                or name.startswith("mars_titan.")
                or name in ("test_context_selection", "test_market_memory")
            ):
                if not Path(module.__file__).resolve().is_relative_to(repository):
                    raise RuntimeError(f"Importación ajena al repositorio: {name}")
        if not torch.cuda.is_available() or "4070" not in torch.cuda.get_device_name(0):
            raise RuntimeError("El protocolo requiere la RTX 4070 en cuda:0")
        torch.cuda.set_device("cuda:0")
        torch.set_num_threads(protocol["cpu_threads"])
        torch.set_num_interop_threads(protocol["interop_threads"])
        torch.backends.mha.set_fastpath_enabled(False)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(protocol["seed"])
        torch.cuda.manual_seed_all(protocol["seed"])
        free, total = torch.cuda.mem_get_info(0)
        if (
            free
            < protocol["torch_allocator_cap_bytes"] + protocol["minimum_global_free_vram_bytes"]
        ):
            raise MemoryError("No queda reserva para la cuota de Torch")
        torch.cuda.set_per_process_memory_fraction(protocol["torch_allocator_cap_bytes"] / total, 0)
        report["environment"] = dict(
            python=sys.version,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(0),
            nvidia_smi=gpu.stdout,
            compute_applications_snapshot=dict(
                returncode=applications.returncode,
                stdout=applications.stdout,
                stderr=applications.stderr,
            ),
            concurrent_workloads="No se controlan ni se detienen otras aplicaciones",
            numerics=dict(
                dtype="torch.float64",
                fastpath=False,
                deterministic=True,
                cpu_threads=torch.get_num_threads(),
            ),
        )
        references = {}
        with torch.no_grad():
            for stage in protocol["stages"]:
                for order_index, order in enumerate(protocol["orders"]):
                    for position, case in enumerate(order):
                        condition_started = time.monotonic()
                        gc.collect()
                        torch.cuda.empty_cache()
                        cpu_rng = torch.get_rng_state().clone()
                        cuda_rng = torch.cuda.get_rng_state(0).clone()
                        topology, batch_size = case["topology"], case["batch"]
                        regional = (
                            None if topology is None else RegionalFusionConfig(topology=topology)
                        )
                        model = test_market_memory.pt4(None, regional_fusion=regional)
                        model.to(device="cuda:0", dtype=torch.float64).eval().requires_grad_(False)
                        assert type(model) is FinancialPredictor and model.config.hidden_size == 32
                        checkpoint = {
                            name: value.detach().clone()
                            if isinstance(value, torch.Tensor)
                            else value
                            for name, value in model.state_dict().items()
                        }
                        checkpoint_proof = fingerprint(torch, checkpoint)
                        initial = batch = engine = None
                        if stage == "load_state_dict":

                            def operation(model=model, checkpoint=checkpoint):
                                return model.load_state_dict(checkpoint, strict=True)

                            def verify(result, model=model, expected=checkpoint_proof):
                                if result.missing_keys or result.unexpected_keys:
                                    raise AssertionError("La carga válida alteró las claves")
                                actual = fingerprint(torch, model.state_dict())
                                if actual != expected:
                                    raise AssertionError("La carga cambió el checkpoint válido")
                                return actual
                        else:
                            engine = FrozenFinancialConsumer(model)
                            flows = tuple(f"US/F{index:04d}" for index in range(batch_size))
                            batch = test_context_selection.batch_on(model, flows, 0)
                            initial = model.initial_state(flows)
                            initial_proof = fingerprint(torch, initial)
                            batch_payload = {
                                field.name: getattr(batch, field.name)
                                for field in fields(batch)
                                if field.name != "_versions"
                            }
                            batch_proof = fingerprint(torch, batch_payload)

                            def operation(engine=engine, batch=batch, initial=initial):
                                return engine.prepare(batch, initial, context_id="a" * 64)

                            def verify(result):
                                return fingerprint(torch, result)

                        record = dict(
                            stage=stage,
                            order=order_index,
                            position=position,
                            **case,
                            checkpoint=checkpoint_proof,
                            cpu_rng_sha256=sha(cpu_rng.numpy().tobytes()),
                            cuda_rng_sha256=sha(cuda_rng.cpu().numpy().tobytes()),
                            parameter_sha256=model._parameter_id,
                            consumer_model_id=engine.model_id if engine is not None else None,
                            resident_parameter_bytes=sum(
                                p.numel() * p.element_size() for p in model.parameters()
                            ),
                        )
                        record.update(measure(torch, operation, protocol, deadline, verify))
                        record["rows_per_second"] = (
                            batch_size * record["calls_per_second"] if stage == "prepare" else None
                        )
                        key = (stage, batch_size, topology)
                        if key in references and references[key] != record["result"]:
                            raise AssertionError("El orden invertido cambió el resultado exacto")
                        references[key] = record["result"]
                        if (
                            fingerprint(torch, checkpoint) != checkpoint_proof
                            or fingerprint(torch, model.state_dict()) != checkpoint_proof
                        ):
                            raise AssertionError("Las medidas cambiaron parámetros o checkpoint")
                        model.verify_parameter_identity()
                        if stage == "prepare":
                            if (
                                fingerprint(torch, initial) != initial_proof
                                or fingerprint(torch, batch_payload) != batch_proof
                            ):
                                raise AssertionError("prepare mutó el lote o estado recibido")
                            engine.verify(strong=True)
                            del batch_payload
                        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
                            cuda_rng, torch.cuda.get_rng_state(0)
                        ):
                            raise AssertionError("La condición consumió el RNG compartido")
                        record.update(status="passed", checkpoint_rng_inputs_unchanged=True)
                        record["condition_total_seconds"] = time.monotonic() - condition_started
                        report["records"].append(record)
                        save(output / "results.json", report)
                        print(
                            f"{args.label} {stage} order={order_index} "
                            f"B={batch_size} {topology}: passed",
                            flush=True,
                        )
                        del model, checkpoint, engine, batch, initial, operation, verify
        if len(report["records"]) != protocol["expected_records_per_version"]:
            raise AssertionError("No se completó la selección fijada")
        if sources(repository) != manifest or PROTOCOL.read_bytes() != protocol_bytes:
            raise AssertionError("Cambió una fuente o el protocolo durante las medidas")
        if sha(Path(__file__).read_bytes()) != report["script_sha256"]:
            raise AssertionError("Cambió el guion durante las medidas")
        report.update(
            status="passed",
            hold_sha256=check_hold(protocol),
            process_rss=rss(),
            source_bytes_preserved=True,
        )
    except BaseException as error:
        report.update(status="failed", error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        report["worker_total_seconds"] = time.monotonic() - started
        report["finished_at_utc"] = datetime.now(UTC).isoformat()
        save(output / "results.json", report)


if __name__ == "__main__":
    main()
