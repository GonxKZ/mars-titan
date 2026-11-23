#!/usr/bin/env python3
"""Comprobación de CUDA MPS en Google Colab y alternativa segura cuando no está permitido.

Verifica de forma limpia:
1. Existencia de `nvidia-cuda-mps-control`, permisos (UID/directorios sin enlaces simbólicos)
   y contexto de espacio de nombres IPC (`/proc/self/ns/ipc`).
2. Estado de GPU 0 vía `nvidia-smi -q -x -i 0`: modo de cómputo (`Default`, `Exclusive_Process`,
   `Prohibited`), estado MIG (`mig_mode/current_mig`, detectando la incompatibilidad de MIG con
   Exclusive_Process)
   y procesos CUDA activos.
3. Ciclo de vida controlado del demonio MPS (`-d`, `get_server_list` compatible con
   `campaign_slots.py:mps_ready`, `get_default_active_thread_percentage`, y parada limpia `quit`).
4. Alternativa segura sin MPS: procesos independientes con compartición temporal CUDA por defecto,
   `bound_vram` por proceso (`budget.cuda_gib`), empaquetado de un único origen en caché de páginas
   y prohibición explícita de mezclar múltiples modelos en CUDA streams dentro del mismo proceso
   Python (preservando FP32 IEEE estricto y `torch.use_deterministic_algorithms(True)`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

DEFAULT_PIPE_DIR = "/tmp/nvidia-mps"
DEFAULT_LOG_DIR = "/tmp/nvidia-log"
MPS_SERVER_NAME = "nvidia-cuda-mps-server"
CUDA_CONTEXT_GIB = 0.5
STRICT_FP32_ENV: dict[str, str] = {
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "TRITON_F32_DEFAULT": "ieee",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
}


def parse_nvidia_smi_mps_context(xml_text: str) -> dict[str, Any]:
    """Extrae GPU, modo de cómputo, estado MIG, memoria y procesos desde `nvidia-smi -q -x -i 0`."""
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError(f"XML inválido de nvidia-smi: {error}") from error
    gpus = root.findall("gpu")
    if len(gpus) != 1:
        raise ValueError(f"Se esperaba 1 GPU en la consulta XML, encontradas {len(gpus)}")
    gpu = gpus[0]

    name = (gpu.findtext("product_name") or "").strip()
    compute_mode = (gpu.findtext("compute_mode") or "").strip()
    mig_raw = (gpu.findtext("mig_mode/current_mig") or "").strip()
    mig_enabled = mig_raw.lower() == "enabled"

    total_text = (gpu.findtext("fb_memory_usage/total") or "").strip()
    free_text = (gpu.findtext("fb_memory_usage/free") or "").strip()
    total_mib = int(total_text.split()[0]) if re.fullmatch(r"\d+ MiB", total_text) else None
    free_mib = int(free_text.split()[0]) if re.fullmatch(r"\d+ MiB", free_text) else None

    mps_server_active = False
    other_compute_pids: list[int] = []
    processes = gpu.find("processes")
    if processes is not None:
        for proc in processes.findall("process_info"):
            pid_s = (proc.findtext("pid") or "").strip()
            kind = (proc.findtext("type") or "").strip()
            pname = (proc.findtext("process_name") or "").strip()
            if not pid_s.isdecimal():
                continue
            base = Path(pname.split()[0]).name if pname else ""
            if base == MPS_SERVER_NAME:
                mps_server_active = True
            elif kind in {"C", "M", "M+C"}:
                other_compute_pids.append(int(pid_s))

    return {
        "gpu_name": name or None,
        "compute_mode": compute_mode or None,
        "mig_raw": mig_raw or None,
        "mig_enabled": mig_enabled,
        "total_mib": total_mib,
        "free_mib": free_mib,
        "mps_server_active": mps_server_active,
        "other_compute_pids": other_compute_pids,
    }


def _validate_directory_not_symlink(path: Path, label: str) -> None:
    if path.is_symlink():
        raise ValueError(f"{label} no puede ser un enlace simbólico: {path}")


def check_mps_prerequisites(
    *,
    which=shutil.which,
    run=subprocess.run,
    geteuid=os.geteuid,
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    ipc_ns_path: Path = Path("/proc/self/ns/ipc"),
) -> dict[str, Any]:
    """Comprueba requisitos de sistema, permisos, GPU, MIG y namespace para CUDA MPS."""
    blockers: list[str] = []
    warnings: list[str] = []

    binary = which("nvidia-cuda-mps-control")
    if not binary:
        blockers.append("Falta el binario nvidia-cuda-mps-control en PATH")

    uid = int(geteuid())
    if uid != 0:
        warnings.append(
            f"Proceso sin privilegios root (uid={uid}): no podrá cambiar compute_mode y "
            "requiere permisos propios en CUDA_MPS_PIPE_DIRECTORY."
        )

    pipe_path = Path(pipe_dir)
    log_path = Path(log_dir)
    for target, label in (
        (pipe_path, "CUDA_MPS_PIPE_DIRECTORY"),
        (log_path, "CUDA_MPS_LOG_DIRECTORY"),
    ):
        if target.is_symlink():
            blockers.append(f"{label} es un enlace simbólico prohibido: {target}")
        else:
            parent = target if target.exists() else target.parent
            if parent.exists() and not os.access(parent, os.W_OK):
                blockers.append(f"{label} no tiene permiso de escritura en {parent}")

    ipc_namespace = None
    try:
        ipc_namespace = os.readlink(ipc_ns_path)
    except OSError:
        warnings.append(
            "No se pudo inspeccionar /proc/self/ns/ipc; verificar aislamiento IPC del contenedor."
        )

    gpu_ctx: dict[str, Any] | None = None
    try:
        res = run(
            ["nvidia-smi", "-q", "-x", "-i", "0"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        gpu_ctx = parse_nvidia_smi_mps_context(res.stdout)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        blockers.append(f"No se pudo consultar la GPU 0 con nvidia-smi: {error}")

    if gpu_ctx is not None:
        cm = gpu_ctx["compute_mode"]
        if cm == "Prohibited":
            blockers.append("La GPU 0 está en modo de cómputo Prohibited (CUDA bloqueado)")
        if gpu_ctx["mig_enabled"] and cm == "Exclusive_Process":
            blockers.append(
                "Conflicto hardware: MIG activo es incompatible con "
                "compute_mode=Exclusive_Process para MPS"
            )
        if gpu_ctx["mig_enabled"]:
            warnings.append(
                "MIG está activo: cada instancia MIG requiere su propio servidor MPS y pipe_dir "
                "aislado."
            )
        if cm == "Default":
            warnings.append(
                "compute_mode=Default: MPS puede operar si el contenedor permite el demonio, pero "
                "los procesos sin CUDA_MPS_PIPE_DIRECTORY compartirán la GPU por turnos fuera de "
                "MPS."
            )
        if gpu_ctx["other_compute_pids"]:
            warnings.append(
                f"Hay procesos CUDA de cómputo activos ({gpu_ctx['other_compute_pids']}) antes de "
                "iniciar MPS."
            )

    return {
        "allowed": len(blockers) == 0,
        "binary": binary,
        "uid": uid,
        "ipc_namespace": ipc_namespace,
        "pipe_dir": str(pipe_path),
        "log_dir": str(log_path),
        "gpu": gpu_ctx,
        "mig_enabled": gpu_ctx["mig_enabled"] if gpu_ctx else False,
        "compute_mode": gpu_ctx["compute_mode"] if gpu_ctx else None,
        "blockers": blockers,
        "warnings": warnings,
    }


def probe_mps_lifecycle(
    *,
    run=subprocess.run,
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
    keep_running: bool = False,
) -> dict[str, Any]:
    """Arranca, verifica (`get_server_list`) y detiene limpiamente el demonio CUDA MPS."""
    pipe_path = Path(pipe_dir)
    log_path = Path(log_dir)
    _validate_directory_not_symlink(pipe_path, "CUDA_MPS_PIPE_DIRECTORY")
    _validate_directory_not_symlink(log_path, "CUDA_MPS_LOG_DIRECTORY")
    pipe_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    log_path.mkdir(mode=0o700, parents=True, exist_ok=True)

    env = {
        **os.environ,
        "CUDA_MPS_PIPE_DIRECTORY": str(pipe_path),
        "CUDA_MPS_LOG_DIRECTORY": str(log_path),
    }
    started = False
    responded = False
    stopped = False
    active_thread_pct: int | None = None
    error_msg: str | None = None

    try:
        start_res = run(
            ["nvidia-cuda-mps-control", "-d"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if start_res.returncode != 0:
            error_msg = (start_res.stderr or start_res.stdout or "fallo al iniciar MPS").strip()
            return {
                "operational": False,
                "started_cleanly": False,
                "responded_to_server_list": False,
                "stopped_cleanly": False,
                "active_thread_pct": None,
                "error": error_msg,
            }
        started = True

        # Contrato idéntico a material/repo-v11/src/mars_titan/training/campaign_slots.py:309-323
        list_res = run(
            ["nvidia-cuda-mps-control"],
            input="get_server_list\n",
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        combined = f"{list_res.stdout}{list_res.stderr}"
        responded = list_res.returncode == 0 and "Cannot find" not in combined
        if not responded:
            error_msg = combined.strip() or "El demonio MPS no respondió a get_server_list"
        else:
            pct_res = run(
                ["nvidia-cuda-mps-control"],
                input="get_default_active_thread_percentage\n",
                env=env,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            token = pct_res.stdout.strip().splitlines()[0].strip() if pct_res.stdout.strip() else ""
            if pct_res.returncode == 0 and token.replace(".", "", 1).isdecimal():
                active_thread_pct = int(float(token))
    except (OSError, subprocess.SubprocessError) as error:
        error_msg = f"{type(error).__name__}: {error}"
    finally:
        if started and (not keep_running or not responded):
            try:
                quit_res = run(
                    ["nvidia-cuda-mps-control"],
                    input="quit\n",
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    check=False,
                )
                stopped = quit_res.returncode == 0
            except (OSError, subprocess.SubprocessError):
                stopped = False

    return {
        "operational": started and responded,
        "started_cleanly": started,
        "responded_to_server_list": responded,
        "stopped_cleanly": stopped,
        "active_thread_pct": active_thread_pct,
        "error": error_msg,
    }


def build_fallback_plan(
    *,
    workers: int,
    vram_total_gib: float,
    host_ram_total_gib: float,
    per_job_vram_gib: float = 12.0,
    per_job_ram_gib: float = 17.0,
    vram_reserve_gib: float = 4.0,
    host_ram_reserve_gib: float = 12.0,
    intra_threads: int = 4,
    mps_blocked_reason: str = "",
) -> dict[str, Any]:
    """Alternativa segura cuando CUDA MPS no está permitido o no mejora el caudal."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError(f"workers debe estar en [1, 8]; recibido {workers!r}")
    if vram_total_gib <= vram_reserve_gib or host_ram_total_gib <= host_ram_reserve_gib:
        raise ValueError("La memoria total debe superar las reservas de seguridad")
    if per_job_vram_gib <= 0 or per_job_ram_gib <= 0:
        raise ValueError("Consumos por trabajo deben ser positivos")

    max_by_vram = max(
        0, int((vram_total_gib - vram_reserve_gib) // (per_job_vram_gib + CUDA_CONTEXT_GIB))
    )
    max_by_ram = max(0, int((host_ram_total_gib - host_ram_reserve_gib) // per_job_ram_gib))
    limits = (("host_ram", max_by_ram), ("vram", max_by_vram), ("requested", workers))
    bottleneck, safe_workers = min(limits, key=lambda pair: pair[1])
    if safe_workers < 1:
        raise ValueError(
            f"No cabe ni 1 trabajo en modo alternativo (max_by_ram={max_by_ram}, "
            f"max_by_vram={max_by_vram})"
        )

    env = dict(STRICT_FP32_ENV)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "OMP_NUM_THREADS": str(intra_threads),
            "OPENBLAS_NUM_THREADS": str(intra_threads),
            "MKL_NUM_THREADS": str(intra_threads),
        }
    )
    return {
        "mode": "fallback_independent_processes",
        "mps_blocked_reason": mps_blocked_reason,
        "primary_strategy": "independent_processes_cuda_default_timeslicing",
        "packing_strategy": "single_origin_page_cache_packing",
        "streams_policy": "intra_process_h2d_prefetch_only",
        "multi_model_same_process_allowed": False,
        "multi_model_same_process_rejection_reason": (
            "Mezclar varios modelos en streams CUDA del mismo proceso Python sufre contención del "
            "GIL "
            "y riesgo de contaminación cruzada de estados RNG/cuBLAS/determinismo; el motor "
            "comparison_campaign.py ya aísla cada trabajo en su proceso con bound_vram()."
        ),
        "safe_workers": safe_workers,
        "max_by_vram": max_by_vram,
        "max_by_ram": max_by_ram,
        "bottleneck": bottleneck,
        "per_process_vram_cap_gib": per_job_vram_gib,
        "environment": env,
        "rationale_from_project": (
            "[Medición del proyecto, dgx-us-campaign.md:247-250] El bucle de Titans está acotado "
            "por "
            "Python/CPU en cada trabajo; procesos independientes sin MPS ya rellenan huecos de GPU "
            "preservando identidad bit a bit."
        ),
    }


def evaluate_mps_or_fallback(
    *,
    workers: int = 4,
    host_ram_total_gib: float = 96.0,
    vram_total_gib: float = 96.0,
    probe_live: bool = False,
    keep_running: bool = False,
    which=shutil.which,
    run=subprocess.run,
    geteuid=os.geteuid,
    pipe_dir: str = DEFAULT_PIPE_DIR,
    log_dir: str = DEFAULT_LOG_DIR,
) -> dict[str, Any]:
    """Evalúa requisitos MPS, ejecuta sonda opcional y emite plan MPS o alternativa segura."""
    prereq = check_mps_prerequisites(
        which=which, run=run, geteuid=geteuid, pipe_dir=pipe_dir, log_dir=log_dir
    )
    gpu_total_gib = (
        prereq["gpu"]["total_mib"] / 1024.0
        if prereq["gpu"] and prereq["gpu"].get("total_mib")
        else vram_total_gib
    )

    lifecycle: dict[str, Any] | None = None
    mps_operational = False
    reason = "; ".join(prereq["blockers"])
    if prereq["allowed"] and probe_live:
        lifecycle = probe_mps_lifecycle(
            run=run, pipe_dir=pipe_dir, log_dir=log_dir, keep_running=keep_running
        )
        mps_operational = bool(lifecycle["operational"])
        if not mps_operational:
            reason = lifecycle.get("error") or "La sonda de ciclo de vida MPS falló"
    elif prereq["allowed"] and not probe_live:
        reason = "Requisitos previos OK pero sonda en vivo no ejecutada (--dry-run)"

    fallback = build_fallback_plan(
        workers=workers,
        vram_total_gib=gpu_total_gib,
        host_ram_total_gib=host_ram_total_gib,
        mps_blocked_reason=reason,
    )

    if mps_operational:
        selected_mode = "mps_enabled"
        active_env = {
            **fallback["environment"],
            "CUDA_MPS_PIPE_DIRECTORY": pipe_dir,
            "CUDA_MPS_LOG_DIRECTORY": log_dir,
            "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT": (
                f"0={int(fallback['per_process_vram_cap_gib'] * 1024)}M"
            ),
        }
    else:
        selected_mode = "fallback_independent_processes"
        active_env = fallback["environment"]

    return {
        "schema_version": 1,
        "kind": "colab_mps_probe_report_v1",
        "selected_mode": selected_mode,
        "mps_operational": mps_operational,
        "fp32_strict_preserved": (
            active_env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
            and active_env.get("TRITON_F32_DEFAULT") == "ieee"
        ),
        "prerequisites": prereq,
        "lifecycle_probe": lifecycle,
        "recommended_workers": fallback["safe_workers"],
        "active_environment": active_env,
        "fallback_plan": fallback,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Comprobación de CUDA MPS en Colab y alternativa segura."
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--host-ram-gib", type=float, default=96.0)
    parser.add_argument("--vram-gib", type=float, default=96.0)
    parser.add_argument("--pipe-dir", default=DEFAULT_PIPE_DIR)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument(
        "--probe-live", action="store_true", help="Prueba arranque/parada real de MPS."
    )
    parser.add_argument(
        "--keep-running", action="store_true", help="Deja MPS activo si la sonda pasa."
    )
    parser.add_argument("--dry-run", action="store_true", help="Evalúa sin lanzar demonio MPS.")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = evaluate_mps_or_fallback(
        workers=args.workers,
        host_ram_total_gib=args.host_ram_gib,
        vram_total_gib=args.vram_gib,
        probe_live=args.probe_live and not args.dry_run,
        keep_running=args.keep_running,
        pipe_dir=args.pipe_dir,
        log_dir=args.log_dir,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
