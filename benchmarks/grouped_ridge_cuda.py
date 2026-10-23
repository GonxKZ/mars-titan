"""Comparar dos resoluciones ridge FP64 sobre sistemas matemáticos de prueba.

Sin argumentos solo comprueba el diseño y las API locales. ``--run`` ejecuta un
hijo CUDA con presupuesto total de 90 segundos. Las entradas son matrices
arbitrarias. No importa el fitter ni construye optimizadores. Las protecciones
de aprendizaje permanecen intactas.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import resource
import signal
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent
SIZES = (3, 32, 512, 4096)
PATTERNS = ("all_observed", "mixed_masks")
FEATURES, OBJECTIVES = 21, 48
REFERENCES = (0, 12)
SEED_BASE = 20261022
MINIMUM, RIDGE = 2, 0.01
WARMUP, REPETITIONS = 3, 7
TOTAL_SECONDS, SOFT_SECONDS = 90, 85
ALLOCATOR_CAP = 512 * 1024**2
FREE_MEMORY_HEADROOM = 256 * 1024**2
AVAILABLE_MEMORY_LIMIT = 1_900_000_000
ATOL, RTOL, STATIONARITY_LIMIT = 1e-12, 3e-11, 1e-10


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def now():
    return datetime.now(UTC).isoformat()


def write_json(path, value):
    if not path.resolve().is_relative_to(BASE):
        raise ValueError("Los recibos deben quedar dentro del área privada")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def plan(*, reverse=False):
    """Comprobar las firmas documentadas sin importar PyTorch o consultar CUDA."""
    sdk = Path(sysconfig.get_paths()["purelib"]) / "torch"
    checks = {
        "linalg/__init__.py": (
            "torch.linalg.lstsq(A, B, rcond=None, *, driver=None)",
            "B (Tensor): rhs tensor of shape `(*, m, k)`",
            "It has shape `(*, n, k)`",
            "For CUDA input, the only valid driver is `'gels'`",
        ),
        "backends/cuda/__init__.py": (
            "def preferred_linalg_library(",
            'If `"cusolver"` is set then cuSOLVER will be used wherever possible.',
        ),
        "cuda/memory.py": (
            "def set_per_process_memory_fraction(",
            "def mem_get_info(",
            "def reset_peak_memory_stats(",
            "def max_memory_allocated(",
            "def max_memory_reserved(",
        ),
    }
    documentation = {}
    for name, expected in checks.items():
        path = sdk / name
        data = path.read_bytes()
        if any(value not in data.decode("utf-8") for value in expected):
            raise RuntimeError(f"La documentación local no confirma la API prevista: {name}")
        documentation[name] = dict(path=str(path), sha256=sha256(data))
    ast.parse(Path(__file__).read_bytes(), filename=str(Path(__file__)))
    return dict(
        kind="mathematical_augmented_ridge_microbenchmark_v1",
        validation_gpu_used=False,
        script_sha256=sha256(Path(__file__).read_bytes()),
        sizes=list(SIZES),
        patterns=list(PATTERNS),
        features=FEATURES,
        augmented_columns=FEATURES + 1,
        objectives=OBJECTIVES,
        reference_columns=list(REFERENCES),
        minimum_observations=MINIMUM,
        ridge=RIDGE,
        seed_base=SEED_BASE,
        warmup_per_variant=WARMUP,
        repetitions_per_variant=REPETITIONS,
        total_budget_seconds=TOTAL_SECONDS,
        soft_budget_seconds=SOFT_SECONDS,
        allocator_cap_bytes=ALLOCATOR_CAP,
        free_memory_headroom_bytes=FREE_MEMORY_HEADROOM,
        available_memory_limit_bytes=AVAILABLE_MEMORY_LIMIT,
        absolute_tolerance=ATOL,
        relative_tolerance=RTOL,
        stationarity_limit=STATIONARITY_LIMIT,
        relative_error_denominator="max(abs(reference), 1e-12)",
        reverse_variant_order=reverse,
        financial_data=False,
        optimizer=False,
        learning_hold_modified=False,
        documentation=documentation,
    )


class BudgetExpired(Exception):
    """Presupuesto matemático agotado antes de iniciar otra operación."""


def check_budget(deadline):
    if time.monotonic() >= deadline:
        raise BudgetExpired("Se agotó el presupuesto técnico. No se inicia otra operación")


def finite(torch, value, name):
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contiene NaN o infinito")


def design(torch, x, weights):
    weights = weights / weights.sum()
    magnitude = x.abs().amax(0).clamp_min(1.0)
    scaled = x / magnitude
    center = (weights[:, None] * scaled).sum(0)
    spread = (weights[:, None] * (scaled - center).square()).sum(0).sqrt()
    lower, upper = x.amin(0), x.amax(0)
    constant = lower == upper
    mean = torch.where(constant, lower, center * magnitude)
    scale = spread * magnitude
    scale = torch.where(constant | (scale == 0), torch.ones_like(scale), scale)
    centered = x - mean
    for value, name in ((mean, "Media"), (scale, "Escala"), (centered, "Centrado")):
        finite(torch, value, name)
    if not bool(((lower <= mean) & (mean <= upper)).all()):
        raise ValueError("La media queda fuera del soporte de las características")
    z = torch.cat((centered / scale, torch.ones_like(x[:, :1])), dim=1)
    finite(torch, z, "Diseño")
    return z, weights


def solve(torch, case, *, grouped, deadline):
    """Mantener el sistema aumentado, sus máscaras y su penalización completos."""
    x, targets, masks, weights = (case[name] for name in ("x", "targets", "masks", "weights"))
    z, weights = design(torch, x, weights)
    finite(torch, targets, "Segundos miembros")
    counts = masks.sum(0).cpu().tolist()
    groups = {}
    for column, count in enumerate(counts):
        if column in REFERENCES or count < MINIMUM:
            continue
        if grouped:
            chosen = case["masks_cpu"][:, column]
            normalized = case["weights_cpu"][chosen]
            normalized = normalized / normalized.sum()
            key = (chosen.numpy().tobytes(), normalized.numpy().tobytes())
        else:
            key = column
        groups.setdefault(key, []).append(column)
    coefficients = z.new_zeros((FEATURES + 1, OBJECTIVES))
    residual_norms = [None] * OBJECTIVES
    conditions = [None] * OBJECTIVES
    penalty = RIDGE**0.5 * torch.eye(z.shape[1], dtype=z.dtype, device=z.device)
    for columns in groups.values():
        check_budget(deadline)
        chosen = masks[:, columns[0]]
        w = weights[chosen]
        w = w / w.sum()
        matrix = torch.cat((w.sqrt()[:, None] * z[chosen], penalty))
        singular = torch.linalg.svdvals(matrix)
        condition = float((singular[0] / singular[-1]).item())
        if not math.isfinite(condition) or condition > 1e12:
            raise ValueError("El sistema matemático está mal condicionado")
        if grouped:
            values = targets[:, columns][chosen]
            magnitude = values.abs().amax(0).clamp_min(1.0)
            rhs = torch.cat(
                (
                    w.sqrt()[:, None] * (values / magnitude[None, :]),
                    z.new_zeros((z.shape[1], len(columns))),
                )
            )
        else:
            values = targets[chosen, columns[0]]
            magnitude = values.abs().amax().clamp_min(1.0)
            rhs = torch.cat((w.sqrt() * (values / magnitude), z.new_zeros(z.shape[1])))
        normalized = torch.linalg.lstsq(matrix, rhs, driver="gels").solution
        residual = matrix.T @ (matrix @ normalized - rhs)
        finite(torch, residual, "Residuo de estacionariedad")
        norms = (
            torch.linalg.vector_norm(residual, dim=0)
            if grouped
            else torch.linalg.vector_norm(residual)
        )
        norm_values = norms.cpu().tolist() if grouped else [float(norms.item())]
        fitted = normalized * magnitude
        finite(torch, fitted, "Coeficientes ridge")
        if grouped:
            coefficients[:, columns] = fitted
        else:
            coefficients[:, columns[0]] = fitted
        for index, column in enumerate(columns):
            residual_norms[column] = norm_values[index]
            conditions[column] = condition
    return dict(
        coefficients=coefficients,
        residual_norms=residual_norms,
        conditions=conditions,
        counts=counts,
        solved_columns=[column for values in groups.values() for column in values],
        factorization_groups=len(groups),
    )


def make_case(torch, size, pattern):
    """Crear solo matrices y segundos miembros arbitrarios con RNG CPU independiente."""
    generator = torch.Generator(device="cpu").manual_seed(SEED_BASE + size)
    x = torch.randn((size, FEATURES), dtype=torch.float64, generator=generator)
    x *= torch.logspace(-2, 2, FEATURES, dtype=torch.float64)
    x[:, -1] = 1.0
    rhs = torch.randn((size, OBJECTIVES), dtype=torch.float64, generator=generator)
    rhs *= torch.logspace(-3, 3, OBJECTIVES, dtype=torch.float64)
    rhs[:, list(REFERENCES)] = 0.0
    rows = torch.arange(size)
    weights = (1 + rows.remainder(5)).double()
    weights /= weights.sum()
    masks = torch.ones((size, OBJECTIVES), dtype=torch.bool)
    if pattern == "mixed_masks":
        patterns = (
            torch.ones(size, dtype=torch.bool),
            rows.remainder(2) == 0,
            rows.remainder(3) != 0,
            rows.remainder(4) != 0,
        )
        for column in range(OBJECTIVES):
            if column not in REFERENCES:
                masks[:, column] = patterns[column % len(patterns)]
        masks[:, 11] = False
        masks[:, 47] = rows == 0
    rhs = torch.where(masks, rhs, torch.zeros_like(rhs))
    cpu = dict(x=x, targets=rhs, masks=masks, weights=weights)
    identities = {name: sha256(value.numpy().tobytes()) for name, value in cpu.items()}
    return dict(
        **{name: value.to("cuda:0") for name, value in cpu.items()},
        masks_cpu=masks,
        weights_cpu=weights,
    ), identities


def percentile(values, quantile):
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def compare(torch, reference, candidate):
    metrics = {}
    passed = True
    if reference["counts"] != candidate["counts"] or set(reference["solved_columns"]) != set(
        candidate["solved_columns"]
    ):
        raise ValueError("Las variantes no resuelven los mismos objetivos observados")
    for name in ("coefficients", "residual_norms"):
        if name == "coefficients":
            expected, actual = reference[name].cpu(), candidate[name].cpu()
        else:
            expected = torch.tensor(
                [0.0 if value is None else value for value in reference[name]], dtype=torch.float64
            )
            actual = torch.tensor(
                [0.0 if value is None else value for value in candidate[name]], dtype=torch.float64
            )
        difference = (actual - expected).abs()
        limit = ATOL + RTOL * expected.abs()
        valid = bool(torch.isfinite(actual).all() and (difference <= limit).all())
        metrics[name] = dict(
            max_absolute_error=float(difference.max()),
            max_relative_error_with_floor=float(
                (difference / expected.abs().clamp_min(ATOL)).max()
            ),
            max_tolerance_ratio=float((difference / limit).max()),
            passed=valid,
        )
        passed = passed and valid
    for label, result in (("reference", reference), ("grouped", candidate)):
        maximum = max(value for value in result["residual_norms"] if value is not None)
        metrics[label + "_max_stationarity_norm"] = maximum
        valid = maximum < STATIONARITY_LIMIT and not bool(
            torch.count_nonzero(result["coefficients"][:, list(REFERENCES)])
        )
        passed = passed and valid
    return dict(passed=passed, errors=metrics)


def run_child(out, *, reverse=False):
    began = time.monotonic()
    deadline = began + SOFT_SECONDS
    receipt = dict(
        plan=plan(reverse=reverse),
        started_at_utc=now(),
        status="incomplete",
        cases=[],
        gpu_used=False,
        reverse_variant_order=reverse,
        timing_scope=(
            "Sistemas con entradas ya residentes en CUDA, normalización, grupos CPU, "
            "SVD, gels, verificaciones y sincronización final. No incluye fuentes financieras"
        ),
        memory_scope=(
            "Picos del allocator por variante, incluidos tres calentamientos y siete repeticiones. "
            "No incluyen toda la memoria del contexto ni asignaciones externas al allocator"
        ),
        counters_scope="Las factorizaciones son llamadas de la implementación, no launches medidos",
        energy_measured=False,
        monetary_cost_measured=False,
    )
    report_path = out / "results.json"
    try:
        import torch

        check_budget(deadline)
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,driver_version",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        applications = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_gpu_memory",
                "--format=csv",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA no está disponible. No hay alternativa CPU")
        torch.cuda.set_device("cuda:0")
        receipt["gpu_used"] = True
        if torch.is_autocast_enabled("cuda"):
            raise RuntimeError("El microbenchmark exige FP64 sin autocast")
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.preferred_linalg_library("cusolver")
        torch.use_deterministic_algorithms(True)
        free, total = torch.cuda.mem_get_info("cuda:0")
        allowed = min(ALLOCATOR_CAP, AVAILABLE_MEMORY_LIMIT, free - FREE_MEMORY_HEADROOM)
        if allowed < 64 * 1024**2:
            raise RuntimeError(f"VRAM libre insuficiente para el límite técnico: {free} bytes")
        torch.cuda.set_per_process_memory_fraction(float(allowed / total), "cuda:0")
        receipt["environment"] = dict(
            python=sys.version,
            executable=sys.executable,
            torch=str(torch.__version__),
            cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name("cuda:0"),
            device="cuda:0",
            dtype="torch.float64",
            autocast_enabled=False,
            nvidia_smi=smi.stdout,
            concurrent_compute_applications=dict(
                exit_code=applications.returncode,
                output=applications.stdout,
                error=applications.stderr,
            ),
            free_vram_before_bytes=free,
            total_vram_bytes=total,
            allocator_limit_bytes=allowed,
            linalg_preference=str(torch.backends.cuda.preferred_linalg_library()),
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cpu_threads=torch.get_num_threads(),
            cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        )
        for size_index, size in enumerate(SIZES):
            for pattern_index, pattern in enumerate(PATTERNS):
                check_budget(deadline)
                prepared_at = time.perf_counter()
                case, identities = make_case(torch, size, pattern)
                torch.cuda.synchronize("cuda:0")
                entry = dict(
                    size=size,
                    pattern=pattern,
                    input_sha256=identities,
                    preparation_seconds=time.perf_counter() - prepared_at,
                    variants={},
                )
                receipt["cases"].append(entry)
                order = (
                    ("reference", "grouped")
                    if (size_index + pattern_index) % 2 == 0
                    else ("grouped", "reference")
                )
                if reverse:
                    order = tuple(reversed(order))
                entry["variant_order"] = list(order)
                outputs = {}
                for name in order:
                    check_budget(deadline)
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats("cuda:0")
                    baseline_allocated = torch.cuda.memory_allocated("cuda:0")
                    baseline_reserved = torch.cuda.memory_reserved("cuda:0")
                    samples = []
                    result = None
                    for iteration in range(WARMUP + REPETITIONS):
                        check_budget(deadline)
                        torch.cuda.synchronize("cuda:0")
                        started = time.perf_counter()
                        with torch.no_grad():
                            result = solve(
                                torch, case, grouped=name == "grouped", deadline=deadline
                            )
                        torch.cuda.synchronize("cuda:0")
                        elapsed = (time.perf_counter() - started) * 1000
                        if iteration >= WARMUP:
                            samples.append(elapsed)
                        if iteration + 1 < WARMUP + REPETITIONS:
                            del result
                    peak_allocated = torch.cuda.max_memory_allocated("cuda:0")
                    peak_reserved = torch.cuda.max_memory_reserved("cuda:0")
                    outputs[name] = {
                        key: value.cpu() if isinstance(value, torch.Tensor) else value
                        for key, value in result.items()
                    }
                    del result
                    entry["variants"][name] = dict(
                        samples_ms=samples,
                        p50_total_ms=statistics.median(samples),
                        p95_total_ms=percentile(samples, 0.95),
                        baseline_allocated_bytes=baseline_allocated,
                        baseline_reserved_bytes=baseline_reserved,
                        peak_allocated_bytes=peak_allocated,
                        peak_reserved_bytes=peak_reserved,
                        peak_allocated_extra_bytes=max(0, peak_allocated - baseline_allocated),
                        peak_reserved_extra_bytes=max(0, peak_reserved - baseline_reserved),
                        factorization_groups=outputs[name]["factorization_groups"],
                        objective_counts=outputs[name]["counts"],
                        solved_columns=sorted(outputs[name]["solved_columns"]),
                        max_augmented_condition=max(
                            value for value in outputs[name]["conditions"] if value is not None
                        ),
                    )
                    write_json(report_path, receipt)
                entry["parity"] = compare(torch, outputs["reference"], outputs["grouped"])
                entry["p50_speedup"] = (
                    entry["variants"]["reference"]["p50_total_ms"]
                    / entry["variants"]["grouped"]["p50_total_ms"]
                )
                entry["p95_speedup"] = (
                    entry["variants"]["reference"]["p95_total_ms"]
                    / entry["variants"]["grouped"]["p95_total_ms"]
                )
                entry["free_vram_after_bytes"] = torch.cuda.mem_get_info("cuda:0")[0]
                write_json(report_path, receipt)
                print(
                    json.dumps(
                        dict(
                            size=size,
                            pattern=pattern,
                            parity=entry["parity"]["passed"],
                            p50_speedup=entry["p50_speedup"],
                        )
                    ),
                    flush=True,
                )
                if not entry["parity"]["passed"]:
                    raise ValueError("La variante agrupada no cumple la paridad fijada")
                del case, outputs
        receipt["status"] = "passed"
    except BaseException as error:
        receipt["error"] = dict(type=type(error).__name__, reason=str(error))
        traceback.print_exc()
    finally:
        receipt["finished_at_utc"] = now()
        receipt["elapsed_seconds"] = time.monotonic() - began
        receipt["peak_process_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        write_json(report_path, receipt)
    return 0 if receipt["status"] == "passed" else 1


def terminate(signum, frame):
    raise SystemExit(128 + signum)


def supervise(label, *, reverse=False):
    if not label or any(not (value.isalnum() or value in "-_") for value in label):
        raise ValueError("La etiqueta solo admite letras, números, guiones y guiones bajos")
    design_record = plan(reverse=reverse)
    out = BASE / label
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
        "--child",
        str(out),
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
        TMPDIR=str(out),
    )
    process, code, timed_out = None, None, False
    began = time.monotonic()
    try:
        with (out / "execution.log").open("x", encoding="utf-8") as stream:
            process = subprocess.Popen(
                command,
                env=environment,
                cwd=out,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(timeout=TOTAL_SECONDS)
            except subprocess.TimeoutExpired:
                timed_out = True
    finally:
        if process is not None and (timed_out or code is None):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            code = process.wait(timeout=5)
        report = dict(
            command=command,
            plan=design_record,
            exit_code=code,
            timed_out=timed_out,
            elapsed_seconds=time.monotonic() - began,
            completed=False,
            reverse_variant_order=reverse,
        )
        results_path = out / "results.json"
        if results_path.exists():
            report["results_sha256"] = sha256(results_path.read_bytes())
            report["completed"] = (
                not timed_out
                and code == 0
                and json.loads(results_path.read_text())["status"] == "passed"
            )
        write_json(out / "execution.json", report)
    print(
        json.dumps(dict(completed=report["completed"], timed_out=timed_out, output=str(out))),
        flush=True,
    )
    return 0 if report["completed"] else 1


def main():
    signal.signal(signal.SIGTERM, terminate)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run", action="store_true", help="Ejecutar el microbenchmark CUDA autorizado"
    )
    parser.add_argument(
        "--label", default="probe-1", help="Directorio nuevo dentro del área privada"
    )
    parser.add_argument(
        "--reverse", action="store_true", help="Invertir el orden de variantes de cada condición"
    )
    parser.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        return run_child(args.child, reverse=args.reverse)
    if args.run:
        return supervise(args.label, reverse=args.reverse)
    print(json.dumps(plan(reverse=args.reverse), ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
