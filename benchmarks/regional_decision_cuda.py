"""Medir la decisión técnica completa con cinco puentes emparejados, sin aprendizaje.

Sin --run imprime el protocolo y no importa Torch. --run crea una salida nueva y
supervisa un único hijo uv durante un máximo de 180 segundos, incluido el arranque.
El hijo usa cuda:0 en FP64 y conserva la guarda global de aprendizaje cerrada.

Se mide FrozenFinancialConsumer.prepare desde antes de su verificación ligera hasta
que devuelve predicción, lectura y propuesta de estado, con sincronización CUDA al
terminar. Incluye los codificadores de las entradas del fixture y MAC. Quedan fuera
la construcción y validación inicial del lote, la codificación del corpus original,
la colección de mecanismos, la persistencia nativa y el feedback. La actualización
asociativa interna de MAC pertenece a prepare. No hay optimizador ni ajuste del
predictor. Cada llamada reutiliza exactamente el mismo estado inicial inmutable.

Las dos pasadas invierten el orden de las cinco variantes dentro de cada forma.
Comparten proceso y GPU con las aplicaciones que ya existan. Diez observaciones
solo dan percentiles descriptivos. RAM es el RSS del proceso y su máximo acumulado.
VRAM es la del asignador de Torch, sin atribuir contexto CUDA o memoria ajena.
Los bytes lógicos no son transferencias físicas ni ancho de banda medido.

Para ejecutar, copiar el guion fuera del repositorio y declarar --repository.
La salida debe ser nueva y quedar junto a esa copia privada.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import resource
import signal
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import fields, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
HOLD = Path.home() / ".local/state/mars-titan/training-hold-2000.json"
HOLD_SHA256 = "8a658080c1ac4e63847eba58b177020339da6ec47021cf26d2701830073c87fd"
REPOSITORY = Path.cwd()
VARIANTS = ("j2", "active-control", "region-removed", "edge-cut", "all-removed")
SHAPES = ((1, 8), (1, 1024), (256, 8), (256, 1024))
WARMUP, REPETITIONS, PROCESS_SECONDS = 3, 10, 180
ALLOCATOR_CAP, HEADROOM = 512 * 1024**2, 256 * 1024**2
DEVICE = "cuda:0"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def protocol():
    """Fijar el alcance y los límites antes de conocer ningún tiempo."""
    return dict(
        kind="frozen_financial_prepare_intervention_fixture_cost_v1",
        script_sha256=sha256(Path(__file__).read_bytes()),
        shapes=[dict(batch=b, episodes=n) for b, n in SHAPES],
        variants=list(VARIANTS),
        orders=[list(VARIANTS), list(reversed(VARIANTS))],
        conditions_distinct=len(SHAPES) * len(VARIANTS),
        condition_records=len(SHAPES) * len(VARIANTS) * 2,
        warmup=WARMUP,
        repetitions=REPETITIONS,
        total_prepare_calls=len(SHAPES) * len(VARIANTS) * 2 * (WARMUP + REPETITIONS),
        predictor_fixture="test_market_memory.pt4(None), seed=42, scalar_head",
        batch_fixture="test_context_selection.batch_on, event=0, seed=23",
        snapshot_fixture="test_episodic_readout.source(N), keys_repeated_to_N_if_N_gt_64",
        snapshot_semantics="synthetic_mature_values_and_labels_not_financial_observations",
        device=DEVICE,
        dtype="torch.float64",
        hidden_size=32,
        regions=3,
        topology="ring",
        neighbors=8,
        K=1,
        J=2,
        cycles=0,
        node_budget_per_row=1216,
        dense_initial_views_per_row=3,
        dense_new_node_updates_per_row=6,
        timed_scope="complete_FrozenFinancialConsumer.prepare_plus_completion_sync",
        includes=[
            "light_verification",
            "fixture_input_encoders",
            "MAC",
            "readout",
            "head",
            "next_state",
        ],
        excludes=[
            "fixture_construction",
            "corpus_encoding",
            "collector",
            "native_persistence",
            "feedback",
            "proof_export",
        ],
        precision_relaxed=False,
        strict_parity="all_active_control_vs_j2_full_result_bitwise_no_tolerance",
        determinism="each_warmup_and_sample_full_result_SHA256_equal_within_variant_and_shape",
        percentiles="linear_interpolation_descriptive_ten_samples_not_stable_tail_estimates",
        throughput="batch_rows_divided_by_mean_synchronized_prepare_seconds",
        allocator_cap_bytes=ALLOCATOR_CAP,
        free_vram_headroom_bytes=HEADROOM,
        headroom_checks="before_and_after_each_prepare_not_continuous_global_monitoring",
        memory_scope="Torch_allocator_and_process_RSS_no_exclusive_variant_VRAM_or_RAM_claim",
        process_budget_seconds=PROCESS_SECONDS,
        optimizer_steps=0,
        shared_predictor_parameter_updates=0,
        real_corpus=False,
        scientific_evaluation=False,
        predictive_improvement_measured=False,
        energy_measured=False,
        physical_transfers_measured=False,
    )


def save(path, value):
    """Confirmar un recibo local sin perder el anterior si falla la escritura."""
    path = Path(path)
    if not path.resolve().is_relative_to(BASE):
        raise ValueError("La salida debe permanecer en el directorio privado del guion")
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


def manifest(root):
    """Sellar fuentes, fixtures y configuración sin consultar ni cambiar Git."""
    result = subprocess.run(
        ["rg", "--files", "src", "tests", "configs", "pyproject.toml", "uv.lock"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
        timeout=8,
    )
    names = sorted(set(result.stdout.splitlines()))
    if not names or len(names) > 4096:
        raise ValueError("El conjunto de fuentes supera el límite técnico")
    total, hashes = 0, {}
    for name in names:
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Una fuente sale del worktree declarado")
        data = path.read_bytes()
        total += len(data)
        if total > 128 * 1024**2:
            raise ValueError("El manifiesto supera 128 MiB de fuentes")
        hashes[name] = sha256(data)
    return hashes


def check_hold():
    data = HOLD.read_bytes()
    if len(data) > 64 * 1024:
        raise ValueError("La guarda global supera el límite de lectura")
    if sha256(data) != HOLD_SHA256 or json.loads(data).get("training_allowed") is not False:
        raise ValueError("La guarda global no coincide o no mantiene cerrado el aprendizaje")
    return sha256(data)


def check_deadline(deadline):
    if time.monotonic() >= deadline:
        raise TimeoutError("Presupuesto de proceso agotado")


def ram():
    status = Path("/proc/self/status").read_text().splitlines()
    rss = next(int(line.split()[1]) * 1024 for line in status if line.startswith("VmRSS:"))
    return dict(
        rss_bytes=rss,
        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )


def nvidia():
    records = {}
    for name, query in (
        (
            "gpu",
            [
                "--id=0",
                "--query-gpu=index,name,uuid,memory.total,memory.free,driver_version",
            ],
        ),
        ("applications", ["--query-compute-apps=pid,process_name,used_gpu_memory"]),
    ):
        result = subprocess.run(
            ["nvidia-smi", *query, "--format=csv"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        records[name] = dict(code=result.returncode, stdout=result.stdout, stderr=result.stderr)
        if name == "gpu" and result.returncode:
            raise RuntimeError("nvidia-smi no pudo verificar la GPU")
    return records


def leaves(torch, value, path="result"):
    """Recorrer el resultado tipado sin omitir pesos rápidos o cursores."""
    if isinstance(value, torch.Tensor):
        yield path, value
    elif is_dataclass(value):
        yield path + "/type", type(value).__module__ + "." + type(value).__qualname__
        for field in fields(value):
            yield from leaves(torch, getattr(value, field.name), path + "/" + field.name)
    elif isinstance(value, dict):
        for name in sorted(value):
            yield from leaves(torch, value[name], path + "/" + str(name))
    elif isinstance(value, (list, tuple)):
        yield path + "/sequence", (type(value).__name__, len(value))
        for index, part in enumerate(value):
            yield from leaves(torch, part, path + "/" + str(index))
    elif value is None or type(value) in (str, int, float, bool):
        yield path, value
    else:
        raise TypeError(f"Campo no previsto en la comprobación completa: {path}")


def proof(torch, value, *, keep=False):
    """Copiar fuera del intervalo medido y exigir finitud y ausencia de grafo."""
    digest, retained, logical = hashlib.sha256(), {}, 0
    for path, item in leaves(torch, value):
        digest.update(json.dumps(path, ensure_ascii=False).encode())
        if isinstance(item, torch.Tensor):
            if item.requires_grad or item.grad_fn is not None or not torch.isfinite(item).all():
                raise AssertionError(f"Tensor no finito o con grafo en {path}")
            cpu = item.detach().to(device="cpu", copy=True).contiguous()
            digest.update(str((tuple(cpu.shape), str(cpu.dtype))).encode())
            digest.update(cpu.numpy().tobytes())
            logical += cpu.numel() * cpu.element_size()
            if keep:
                retained[path] = cpu
        else:
            digest.update(json.dumps(item, ensure_ascii=False, allow_nan=False).encode())
            if keep:
                retained[path] = item
    return digest.hexdigest(), logical, retained


def strict_equal(torch, left, right):
    if left.keys() != right.keys():
        raise AssertionError("El control activo cambió los campos del resultado")
    for path, expected in left.items():
        actual = right[path]
        if isinstance(expected, torch.Tensor):
            if (
                not isinstance(actual, torch.Tensor)
                or actual.dtype != expected.dtype
                or actual.shape != expected.shape
                or not torch.equal(actual, expected)
            ):
                raise AssertionError(f"El control activo perdió paridad exacta en {path}")
        elif type(actual) is not type(expected) or actual != expected:
            raise AssertionError(f"El control activo cambió metadatos en {path}")


def percentile(samples, quantile):
    ordered = sorted(samples)
    position = (len(ordered) - 1) * quantile
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def check_memory(torch):
    allocated = torch.cuda.memory_allocated(DEVICE)
    reserved = torch.cuda.memory_reserved(DEVICE)
    peak_a = torch.cuda.max_memory_allocated(DEVICE)
    peak_r = torch.cuda.max_memory_reserved(DEVICE)
    free, total = torch.cuda.mem_get_info(DEVICE)
    if max(allocated, reserved, peak_a, peak_r) > ALLOCATOR_CAP:
        raise MemoryError("El asignador superó 512 MiB")
    if free < HEADROOM:
        raise MemoryError("La VRAM libre quedó por debajo de 256 MiB")
    return dict(
        allocated_bytes=allocated,
        reserved_bytes=reserved,
        peak_allocated_bytes=peak_a,
        peak_reserved_bytes=peak_r,
        free_global_vram_bytes=free,
        total_global_vram_bytes=total,
    )


def measure_shape(torch, out, report, model, engines, batch_size, episodes, deadline):
    """Reutilizar exactamente lote, evidencia y estado en ambos órdenes."""
    from test_context_selection import batch_on
    from test_episodic_readout import CODEC, CONTEXT, source

    from mars_titan.models.titans.episodic_snapshot import EpisodeSnapshot

    check_deadline(deadline)
    flows = tuple(f"US/F{i:04d}" for i in range(batch_size))
    batch = batch_on(model, flows, 0)
    initial = model.initial_state(flows)
    raw = source(episodes)
    if len(raw["keys"]) != episodes:
        raw["keys"] = raw["keys"].repeat((math.ceil(episodes / len(raw["keys"])), 1))[:episodes]
    snapshot = EpisodeSnapshot.create(
        **raw,
        codec_id=CODEC,
        context_id=CONTEXT,
        cutoff=batch.prediction_at[0],
        device=DEVICE,
        dtype=torch.float64,
    )
    batch_payload = {
        field.name: getattr(batch, field.name)
        for field in fields(batch)
        if field.name != "_versions"
    }
    initial_sha, initial_bytes, _ = proof(torch, initial)
    batch_sha, batch_bytes, _ = proof(torch, batch_payload)
    snapshot_before = snapshot.export_cpu()
    snapshot_sha, snapshot_bytes, _ = proof(torch, snapshot_before)
    cpu_rng, cuda_rng = torch.random.get_rng_state(), torch.cuda.get_rng_state(0)
    entry = dict(
        batch=batch_size,
        episodes=episodes,
        snapshot_sha256=snapshot._digest,
        initial_state_sha256=initial_sha,
        batch_sha256=batch_sha,
        logical_tensor_bytes=dict(
            batch=batch_bytes, initial_state=initial_bytes, snapshot=snapshot_bytes
        ),
        records=[],
    )
    report["cases"].append(entry)
    references, baseline_cpu, common_ids = {}, None, None
    for pass_index in range(2):
        order = list(VARIANTS if pass_index == 0 else reversed(VARIANTS))
        for position, variant in enumerate(order):
            check_deadline(deadline)
            engine = engines[variant]
            record = dict(
                variant=variant,
                pass_index=pass_index,
                reverse=bool(pass_index),
                position=position,
                actual_order=order,
                status="incomplete",
                warmup_completed=0,
                samples=[],
            )
            entry["records"].append(record)
            record["ram_before"] = ram()
            for iteration in range(WARMUP + REPETITIONS):
                check_deadline(deadline)
                check_memory(torch)
                torch.cuda.synchronize(DEVICE)
                torch.cuda.reset_peak_memory_stats(DEVICE)
                before_a = torch.cuda.memory_allocated(DEVICE)
                before_r = torch.cuda.memory_reserved(DEVICE)
                start = time.perf_counter()
                result = engine.prepare(batch, initial, context_id=CONTEXT, snapshot=snapshot)
                torch.cuda.synchronize(DEVICE)
                elapsed = time.perf_counter() - start
                memory = check_memory(torch)
                kept = variant in ("j2", "active-control") and iteration == 0
                result_sha, result_bytes, cpu = proof(torch, result, keep=kept)
                if variant not in references:
                    references[variant] = result_sha
                if result_sha != references[variant]:
                    raise AssertionError("Una repetición cambió el resultado completo")
                if variant == "j2" and baseline_cpu is None:
                    baseline_cpu = cpu
                if variant == "active-control" and kept:
                    strict_equal(torch, baseline_cpu, cpu)
                    if result_sha != references["j2"]:
                        raise AssertionError("El control activo cambió los bytes del resultado")
                read = result.readout.reads[0]
                ids = read.bridge.evidence_ids.detach().cpu()
                if common_ids is None:
                    common_ids = ids.clone()
                if ids.shape != (batch_size, 8) or not torch.equal(ids, common_ids):
                    raise AssertionError("Las intervenciones cambiaron los IDs comunes en K1")
                if (
                    read.bridge.snapshot_sha256 != snapshot._digest
                    or len(result.readout.reads) != 1
                ):
                    raise AssertionError("La lectura perdió el episodio común o K1")
                if not torch.equal(
                    result.next_state.observed_steps,
                    torch.ones(batch_size, dtype=torch.int64),
                ) or not torch.equal(
                    result.next_state.mac.memory.steps.cpu(),
                    torch.ones(batch_size, dtype=torch.int64),
                ):
                    raise AssertionError("prepare no realizó exactamente una observación MAC")
                if variant == "all-removed" and (
                    read.presence.any()
                    or read.values.count_nonzero()
                    or read.bridge.evidence_weights.count_nonzero()
                    or read.bridge.region_values.count_nonzero()
                ):
                    raise AssertionError("La retirada total conservó acceso regional útil")
                if variant == "region-removed" and (
                    read.bridge.evidence_weights[:, 1].count_nonzero()
                    or read.bridge.region_values[:, 1].count_nonzero()
                ):
                    raise AssertionError("La región retirada conserva peso o estado")
                if variant == "edge-cut" and read.bridge.bridge_weights[:, 0, 1].count_nonzero():
                    raise AssertionError("El corte dirigido no se aplicó")
                record["result_sha256"] = result_sha
                record["logical_result_tensor_bytes"] = result_bytes
                if iteration < WARMUP:
                    record["warmup_completed"] += 1
                else:
                    record["samples"].append(
                        dict(
                            seconds=elapsed,
                            **memory,
                            peak_allocated_extra_bytes=memory["peak_allocated_bytes"] - before_a,
                            peak_reserved_extra_bytes=memory["peak_reserved_bytes"] - before_r,
                        )
                    )
                report["allocator_peak_allocated_bytes"] = max(
                    report.get("allocator_peak_allocated_bytes", 0),
                    memory["peak_allocated_bytes"],
                )
                report["allocator_peak_reserved_bytes"] = max(
                    report.get("allocator_peak_reserved_bytes", 0),
                    memory["peak_reserved_bytes"],
                )
                del result, cpu, ids, read
            seconds = [sample["seconds"] for sample in record["samples"]]
            record.update(
                status="passed",
                p50_ms=1000 * statistics.median(seconds),
                p95_ms=1000 * percentile(seconds, 0.95),
                p99_ms=1000 * percentile(seconds, 0.99),
                mean_ms=1000 * statistics.mean(seconds),
                sample_stdev_ms=1000 * statistics.stdev(seconds),
                rows_per_second=batch_size / statistics.mean(seconds),
                ram_after=ram(),
            )
            save(out / "measurement.json", report)
            print(
                f"B={batch_size} N={episodes} pass={pass_index} variant={variant}: passed",
                flush=True,
            )
    if (
        proof(torch, initial)[0] != initial_sha
        or proof(torch, batch_payload)[0] != batch_sha
        or proof(torch, snapshot.export_cpu())[0] != snapshot_sha
    ):
        raise AssertionError("El estado inicial, las filas o el snapshot fueron mutados")
    if not torch.equal(cpu_rng, torch.random.get_rng_state()) or not torch.equal(
        cuda_rng, torch.cuda.get_rng_state(0)
    ):
        raise AssertionError("prepare consumió el RNG compartido")
    for engine in engines.values():
        engine.verify(strong=True)
    entry.update(
        strict_active_control_parity=True,
        deterministic_repetitions=True,
        common_ids_sha256=proof(torch, common_ids)[0],
        common_ids_first_row=common_ids[0].tolist(),
        initial_state_batch_snapshot_rng_unchanged=True,
    )
    del baseline_cpu, initial, batch, batch_payload, snapshot, snapshot_before, raw
    gc.collect()
    torch.cuda.empty_cache()
    save(out / "measurement.json", report)


def worker(root, out, deadline):
    started = time.monotonic()
    deadline = min(deadline, started + PROCESS_SECONDS - 3)
    report = dict(protocol=protocol(), status="incomplete", started_at_utc=now(), cases=[])
    try:
        pinned = json.loads((out / "source-manifest.json").read_text())
        if manifest(root) != pinned:
            raise ValueError("Las fuentes cambiaron antes de importar el producto")
        check_hold()
        check_deadline(deadline)
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. No se admite sustitución por CPU")
        torch.cuda.set_device(DEVICE)
        if "4070" not in torch.cuda.get_device_name(0):
            raise RuntimeError("cuda:0 no es la RTX 4070 prevista")
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.mha.set_fastpath_enabled(False)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if torch.is_autocast_enabled("cuda") or torch.is_inference_mode_enabled():
            raise ValueError("El recorrido requiere FP64 explícito fuera de inference_mode")
        free, total = torch.cuda.mem_get_info(DEVICE)
        if free < ALLOCATOR_CAP + HEADROOM:
            raise MemoryError("No queda reserva para 512 MiB de asignador y 256 MiB libres")
        torch.cuda.set_per_process_memory_fraction(float(ALLOCATOR_CAP / total), DEVICE)
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
            free_vram_before_bytes=free,
            total_vram_bytes=total,
            allocator_fraction=torch.cuda.get_per_process_memory_fraction(DEVICE),
            cuda_memory_api_source_sha256=sha256(Path(torch.cuda.memory.__file__).read_bytes()),
            cpu_threads=torch.get_num_threads(),
            interop_threads=torch.get_num_interop_threads(),
            cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            tf32=False,
            autocast=False,
            deterministic_algorithms=True,
            mha_fastpath=False,
        )
        for path in (
            root / "src",
            root,
            root / "tests/models/titans",
            root / "tests/memory",
        ):
            sys.path.insert(0, str(path))
        from test_bridge_episodic_readout import configuration
        from test_market_memory import pt4

        from mars_titan.models.titans.bridge_intervention import (
            BridgeInterventionConfig,
        )
        from mars_titan.models.titans.bridge_recurrence import BridgeRecurrenceConfig
        from mars_titan.models.titans.episodic_readout import EpisodicReadout
        from mars_titan.models.titans.frozen_financial import FrozenFinancialConsumer

        model = pt4(None).to(DEVICE).eval().requires_grad_(False)
        engines, parameter_hashes = {}, None
        for variant in VARIANTS:
            check_deadline(deadline)
            intervention = (
                None
                if variant == "j2"
                else BridgeInterventionConfig(
                    tuple(
                        variant != "all-removed" and not (variant == "region-removed" and i == 1)
                        for i in range(3)
                    ),
                    ((0, 1),) if variant == "edge-cut" else (),
                )
            )
            base = configuration()
            config = replace(
                base,
                evidence_bridge=replace(
                    base.evidence_bridge,
                    cycles=0,
                    recurrence=BridgeRecurrenceConfig(2, 1216),
                    intervention=intervention,
                ),
            )
            reader = (
                EpisodicReadout(config, device=DEVICE, dtype=torch.float64)
                .eval()
                .requires_grad_(False)
            )
            hashes = {name: proof(torch, value)[0] for name, value in reader.named_parameters()}
            if parameter_hashes is None:
                parameter_hashes = hashes
            if hashes != parameter_hashes:
                raise AssertionError("Los lectores no parten de parámetros emparejados")
            engines[variant] = FrozenFinancialConsumer(model, readout=reader)
        if len({engine.model_id for engine in engines.values()}) != len(VARIANTS):
            raise AssertionError("Las condiciones activas no tienen identidades distintas")
        report.update(
            readout_parameter_sha256=parameter_hashes,
            predictor_parameter_sha256=model._parameter_id,
            resident_model_logical_tensor_bytes=dict(
                common_predictor=sum(
                    value.numel() * value.element_size()
                    for value in (*model.parameters(), *model.buffers())
                ),
                readers={
                    name: sum(
                        value.numel() * value.element_size()
                        for value in (*engine.readout.parameters(), *engine.readout.buffers())
                    )
                    for name, engine in engines.items()
                },
            ),
            parameter_pairing="same_seed_and_exact_named_parameter_hashes_no_cross_identity_copy",
            shared_predictor_object=True,
            consumer_identities={name: engine.identity() for name, engine in engines.items()},
            model_ids={name: engine.model_id for name, engine in engines.items()},
            initialization_outside_timed_scope=True,
        )
        save(out / "measurement.json", report)
        for batch_size, episodes in SHAPES:
            measure_shape(torch, out, report, model, engines, batch_size, episodes, deadline)
        if manifest(root) != pinned:
            raise ValueError("Las fuentes cambiaron durante la medida")
        report.update(
            status="passed",
            hold_sha256_final=check_hold(),
            source_manifest_unchanged=True,
            actual_condition_records=sum(len(case["records"]) for case in report["cases"]),
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
        report.update(
            finished_at_utc=now(),
            child_process_seconds=time.monotonic() - started,
            ram_final=ram(),
        )
        save(out / "measurement.json", report)


def supervise(root, out):
    """Limitar el grupo propio y conservar intentos fallidos o incompletos."""
    started, deadline = time.monotonic(), time.monotonic() + PROCESS_SECONDS
    if not out.resolve().is_relative_to(BASE) or out.exists():
        raise ValueError("--output debe ser una ruta privada nueva bajo el guion")
    out.mkdir(parents=True, exist_ok=False)
    receipt = dict(
        status="incomplete",
        started_at_utc=now(),
        protocol=protocol(),
        repository=str(root),
        output=str(out),
        hold_before=None,
        child_started=False,
    )
    process = None
    try:
        receipt["hold_before"] = check_hold()
        save(out / "protocol.json", protocol())
        save(out / "source-manifest.json", manifest(root))
        receipt["hardware_before"] = nvidia()
        check_deadline(deadline)
        environment = dict(os.environ)
        environment.update(
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            OPENBLAS_NUM_THREADS="2",
            MARS_TITAN_REQUIRE_CUDA="1",
            PYTHONDONTWRITEBYTECODE="1",
        )
        environment["PYTHONPATH"] = os.pathsep.join(
            str(root / part) for part in ("src", ".", "tests/models/titans", "tests/memory")
        )
        command = [
            "uv",
            "run",
            "--no-project",
            "--python",
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--repository",
            str(root),
            "--output",
            str(out),
            "--deadline",
            repr(deadline - 3),
        ]
        receipt["command"] = command
        save(out / "execution.json", receipt)
        with (
            (out / "stdout.log").open("w") as stdout,
            (out / "stderr.log").open("w") as stderr,
        ):
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
                receipt["timeout"] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=0.5)
                raise TimeoutError("El supervisor agotó el límite de 180 segundos") from error
        receipt["returncode"] = process.returncode
        measurement = json.loads((out / "measurement.json").read_text())
        if process.returncode != 0 or measurement["status"] != "passed":
            raise RuntimeError("El hijo no completó todas las condiciones")
        if measurement.get("actual_condition_records") != protocol()["condition_records"]:
            raise AssertionError("Faltan condiciones completas")
        if manifest(root) != json.loads((out / "source-manifest.json").read_text()):
            raise ValueError("Cambió el sello de fuentes")
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
            receipt["hold_after_sha256"] = sha256(HOLD.read_bytes())
        except OSError as error:
            receipt["hold_after_sha256"] = None
            receipt["hold_after_error"] = str(error)
        receipt["hold_unchanged"] = receipt["hold_after_sha256"] == receipt["hold_before"]
        if not receipt["hold_unchanged"]:
            receipt["status"] = "failed"
        receipt.update(
            finished_at_utc=now(),
            total_supervised_process_seconds=time.monotonic() - started,
        )
        if receipt["total_supervised_process_seconds"] > PROCESS_SECONDS:
            receipt["status"] = "incomplete"
            receipt["budget_exceeded"] = True
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
        raise RuntimeError("El cierre no confirmó todas las guardas del benchmark")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repository", type=Path, default=REPOSITORY)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--deadline", type=float, help=argparse.SUPPRESS)
    options = parser.parse_args()
    root = options.repository.resolve(strict=True)
    if options.worker:
        if options.run or options.output is None or options.deadline is None:
            parser.error("El hijo necesita salida y plazo del supervisor")
        worker(root, options.output.resolve(), options.deadline)
    elif options.run:
        if options.output is None:
            parser.error("--run necesita una salida privada nueva mediante --output")
        supervise(root, options.output.resolve())
    else:
        print(json.dumps(protocol(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
