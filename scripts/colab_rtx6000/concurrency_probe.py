#!/usr/bin/env python3
"""Sonda de concurrencia (1, 2, 4 y 6 trabajos) sobre telemetría real de Colab RTX PRO 6000.

Analiza RAM del anfitrión, VRAM de la GPU y pasos por minuto leyendo los registros reales
que genera el motor de campaña en `material/repo-v11`:
- `queue/<modelo>--<año>--seed-<semilla>.json` (`comparison_campaign.py` y
  `comparison_telemetry.py`)
- `runs/<modelo>/<año>/seed-<semilla>/fit/run.json` (`financial_run.py`)
- `equipo.json` (`colab_site.py`)
- `samples.jsonl` (muestreo opcional en vivo de `/proc/meminfo` y `nvidia-smi`).

Ofrece modo de planificación/ejecución en Colab (`plan` / `run-live`) y modo de análisis
offline en CPU (`analyze`), respetando FP32 IEEE estricto y el sellado absoluto de 2024.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROBE_LEVELS: tuple[int, ...] = (1, 2, 4, 6)
FORBIDDEN_YEAR = 2024
CUDA_CONTEXT_MIB = 512
MIB = 1024**2
GIB = 1024**3
OOM_EXIT_CODES = frozenset({137, -9})
OOM_ERROR_TYPES = frozenset({"OutOfMemoryError", "CUDAOutOfMemoryError", "MemoryError"})
_SITE_RE = re.compile(r"^[a-z0-9][a-z0-9.-]{0,63}$")
_YEAR_KEYS = frozenset(
    {
        "year",
        "years",
        "origin",
        "origins",
        "origin_year",
        "origin_years",
        "año",
        "años",
        "origen",
        "origenes",
        "orígenes",
    }
)
_FINAL_TEST_KEYS = frozenset({"final_test_access", "final_test_opened"})
_FORBIDDEN_STR_RE = re.compile(
    r"(?:^|[^0-9])2024(?:[^0-9]|$)|2024[-/]?(?:0[1-9]|1[0-2])[-/]?(?:0[1-9]|[12][0-9]|3[01])|fold-019|final[_-]test",
    re.IGNORECASE,
)


def _check_year_value(value: Any, label: str) -> None:
    if isinstance(value, bool):
        raise ValueError(f"Año inválido booleano en {label}")
    if isinstance(value, int) and value >= FORBIDDEN_YEAR:
        raise ValueError(f"Origen {value} sellado (>= {FORBIDDEN_YEAR}) en {label}")
    if (
        isinstance(value, str)
        and value.strip().isdecimal()
        and int(value.strip()) >= FORBIDDEN_YEAR
    ):
        raise ValueError(f"Origen {value!r} sellado (>= {FORBIDDEN_YEAR}) en {label}")


def assert_no_sealed_2024(obj: Any, *, context: str = "") -> None:
    """Bloquea cualquier referencia al año sellado 2024, `fold-019` o apertura de `final_test`."""
    if isinstance(obj, Path):
        assert_no_sealed_2024(str(obj), context=context)
        return
    if isinstance(obj, str):
        if _FORBIDDEN_STR_RE.search(obj):
            raise ValueError(f"Referencia prohibida al periodo sellado 2024 ({context or obj!r})")
        return
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            key_lower = key_str.lower()
            label = f"{context}.{key_str}" if context else key_str
            if key_lower in _FINAL_TEST_KEYS:
                if value is not False:
                    raise ValueError(
                        f"Acceso prohibido a final_test ({key_str}={value!r}) en "
                        f"{context or 'documento'}"
                    )
                continue
            if key_lower in _YEAR_KEYS:
                if isinstance(value, (list, tuple, set)):
                    for item in value:
                        _check_year_value(item, label)
                else:
                    _check_year_value(value, label)
            assert_no_sealed_2024(key_str, context=label)
            assert_no_sealed_2024(value, context=label)
        return
    if isinstance(obj, (list, tuple, set)):
        for idx, item in enumerate(obj):
            assert_no_sealed_2024(item, context=f"{context}[{idx}]")


def parse_meminfo(text: str) -> dict[str, int]:
    """Extrae contadores de `/proc/meminfo` en KiB y calcula memoria usada."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        name, sep, rest = line.partition(":")
        if not sep:
            continue
        name = name.strip()
        if name in {"MemTotal", "MemAvailable", "MemFree", "SwapTotal", "SwapFree"}:
            parts = rest.split()
            if parts and parts[0].isdecimal():
                values[name] = int(parts[0])
    if "MemTotal" not in values or "MemAvailable" not in values:
        raise ValueError("Faltan MemTotal o MemAvailable en /proc/meminfo")
    if values["MemTotal"] <= 0 or values["MemAvailable"] > values["MemTotal"]:
        raise ValueError("Contadores incoherentes en /proc/meminfo")
    values["used_kib"] = values["MemTotal"] - values["MemAvailable"]
    return values


def parse_nvidia_smi_memory(csv_text: str) -> dict[str, int]:
    """Parsea `nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu`.

    El formato pedido es `csv,noheader,nounits`.
    """
    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("Salida vacía de nvidia-smi")
    fields = [part.strip() for part in lines[0].split(",")]
    if len(fields) != 3 or not all(field.isdecimal() for field in fields):
        raise ValueError(f"Formato inválido de nvidia-smi: {lines[0]!r}")
    used_mib, total_mib, util_pct = (int(field) for field in fields)
    if total_mib <= 0 or not (0 <= used_mib <= total_mib) or not (0 <= util_pct <= 100):
        raise ValueError(f"Valores fuera de rango en nvidia-smi: {fields}")
    return {"used_mib": used_mib, "total_mib": total_mib, "util_pct": util_pct}


def load_site_sheet(path: Path) -> dict[str, Any]:
    """Carga `equipo.json` generado por `colab_site.py` y verifica el contrato FP32."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"La ficha del equipo no es un archivo regular: {path}")
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("kind") != "machine_site_v1":
        raise ValueError(f"Ficha de equipo incompatible: kind={doc.get('kind')!r}")
    assert_no_sealed_2024(doc, context=str(path))

    mem_kib = doc.get("memory_kib")
    host_ram_total_gib = (
        (float(mem_kib) / (1024.0 * 1024.0)) if isinstance(mem_kib, int) and mem_kib > 0 else None
    )

    gpu = doc.get("gpu") or {}
    mem_mib = gpu.get("memory_mib")
    gpu_vram_total_gib = (
        (float(mem_mib) / 1024.0) if isinstance(mem_mib, int) and mem_mib > 0 else None
    )

    torch_info = doc.get("torch")
    fp32_strict_ok = True
    if isinstance(torch_info, dict):
        fp32_strict_ok = (
            torch_info.get("allow_tf32") is False
            and torch_info.get("matmul_precision") == "highest"
        )

    return {
        "site": doc.get("site"),
        "gpu_name": gpu.get("name"),
        "gpu_capability": gpu.get("capability"),
        "gpu_driver": gpu.get("driver"),
        "host_ram_total_gib": host_ram_total_gib,
        "gpu_vram_total_gib": gpu_vram_total_gib,
        "cpu_logical": (doc.get("cpu") or {}).get("logical"),
        "disk_free_gib": (
            (doc["disk"]["free_bytes"] / float(GIB))
            if isinstance(doc.get("disk"), dict) and isinstance(doc["disk"].get("free_bytes"), int)
            else None
        ),
        "fp32_strict_ok": fp32_strict_ok,
        "raw": doc,
    }


def _is_oom(attempt: dict[str, Any]) -> bool:
    code = attempt.get("process_exit_code")
    if isinstance(code, int) and code in OOM_EXIT_CODES:
        return True
    err_type = attempt.get("error_type")
    if isinstance(err_type, str) and err_type in OOM_ERROR_TYPES:
        return True
    msg = str(attempt.get("error") or "").lower()
    return "out of memory" in msg or "cannot allocate memory" in msg or "cuda oom" in msg


def _extract_steps(queue_record: dict[str, Any], fit_report: dict[str, Any] | None) -> int:
    if isinstance(fit_report, dict):
        if isinstance(fit_report.get("global_step"), int) and fit_report["global_step"] >= 0:
            return int(fit_report["global_step"])
        history = fit_report.get("history")
        if isinstance(history, list):
            total = 0
            for item in history:
                train = item.get("train") if isinstance(item, dict) else None
                if isinstance(train, dict) and isinstance(train.get("updates"), int):
                    total += int(train["updates"])
            if total > 0:
                return total
    result = queue_record.get("result")
    if isinstance(result, dict) and isinstance(result.get("global_step"), int):
        return int(result["global_step"])
    return 0


def _extract_wall_seconds(attempt: dict[str, Any], fit_report: dict[str, Any] | None) -> float:
    resources = attempt.get("resources") if isinstance(attempt.get("resources"), dict) else {}
    for candidate in (
        resources.get("wall_seconds"),
        attempt.get("seconds"),
        attempt.get("process_wall_seconds"),
    ):
        if (
            isinstance(candidate, (int, float))
            and not isinstance(candidate, bool)
            and math.isfinite(candidate)
            and candidate > 0
        ):
            return float(candidate)
    if isinstance(fit_report, dict) and isinstance(fit_report.get("attempts"), list):
        total = sum(
            float(a["seconds"])
            for a in fit_report["attempts"]
            if isinstance(a, dict)
            and isinstance(a.get("seconds"), (int, float))
            and not isinstance(a.get("seconds"), bool)
            and a["seconds"] > 0
        )
        if total > 0:
            return total
    return 0.0


def extract_job_telemetry(
    queue_record: dict[str, Any],
    fit_report: dict[str, Any] | None = None,
    *,
    job_key: str = "",
) -> dict[str, Any]:
    """Extrae métricas de RAM, VRAM, OOM y pasos/min de un trabajo real de `comparison_campaign`."""
    assert_no_sealed_2024(job_key, context="job_key")
    assert_no_sealed_2024(queue_record, context=job_key or "queue_record")
    if fit_report is not None:
        assert_no_sealed_2024(fit_report, context=f"{job_key}:fit_report")

    attempts = queue_record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise ValueError(f"El registro de cola {job_key or '<sin clave>'} no tiene intentos")
    last = attempts[-1]
    resources = last.get("resources") if isinstance(last.get("resources"), dict) else {}

    steps = _extract_steps(queue_record, fit_report)
    wall_s = _extract_wall_seconds(last, fit_report)
    if steps > 0 and wall_s <= 0.0:
        raise ValueError(f"wall_seconds inválido ({wall_s}) con steps={steps} en {job_key}")
    steps_per_min = (steps / (wall_s / 60.0)) if (steps > 0 and wall_s > 0.0) else 0.0

    rss_b = resources.get("peak_rss_bytes")
    vram_alloc_b = resources.get("cuda_allocated_peak_bytes")
    vram_res_b = resources.get("cuda_reserved_peak_bytes")
    vram_sampled_b = resources.get("gpu_memory_peak_sampled_bytes")

    return {
        "job_key": job_key,
        "model": queue_record.get("model"),
        "year": queue_record.get("year"),
        "seed": queue_record.get("seed"),
        "status": queue_record.get("status"),
        "attempt_status": last.get("status"),
        "oom_detected": _is_oom(last),
        "steps": steps,
        "wall_seconds": wall_s,
        "steps_per_min": steps_per_min,
        "peak_rss_gib": (float(rss_b) / float(GIB))
        if isinstance(rss_b, int) and rss_b >= 0
        else None,
        "vram_allocated_gib": (
            (float(vram_alloc_b) / float(GIB))
            if isinstance(vram_alloc_b, int) and vram_alloc_b >= 0
            else None
        ),
        "vram_reserved_gib": (
            (float(vram_res_b) / float(GIB))
            if isinstance(vram_res_b, int) and vram_res_b >= 0
            else None
        ),
        "vram_sampled_gib": (
            (float(vram_sampled_b) / float(GIB))
            if isinstance(vram_sampled_b, int) and vram_sampled_b >= 0
            else None
        ),
        "site": resources.get("site"),
    }


def _read_samples_jsonl(samples_path: Path | None) -> dict[str, Any]:
    if samples_path is None or not samples_path.exists():
        return {
            "count": 0,
            "peak_host_ram_gib": None,
            "peak_gpu_vram_gib": None,
            "mean_gpu_util_pct": None,
        }
    if samples_path.is_symlink() or not samples_path.is_file():
        raise ValueError(f"Archivo de muestras no regular o enlace simbólico: {samples_path}")
    ram_vals: list[float] = []
    vram_vals: list[float] = []
    util_vals: list[float] = []
    for raw in samples_path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if isinstance(row.get("host_ram_used_gib"), (int, float)):
            ram_vals.append(float(row["host_ram_used_gib"]))
        if isinstance(row.get("gpu_mem_used_mib"), (int, float)):
            vram_vals.append(float(row["gpu_mem_used_mib"]) / 1024.0)
        if isinstance(row.get("gpu_util_pct"), (int, float)):
            util_vals.append(float(row["gpu_util_pct"]))
    return {
        "count": max(len(ram_vals), len(vram_vals)),
        "peak_host_ram_gib": max(ram_vals) if ram_vals else None,
        "peak_gpu_vram_gib": max(vram_vals) if vram_vals else None,
        "mean_gpu_util_pct": statistics.fmean(util_vals) if util_vals else None,
    }


def analyze_level_state(
    state_dir: Path,
    *,
    workers: int,
    site_sheet: dict[str, Any] | None = None,
    samples_path: Path | None = None,
) -> dict[str, Any]:
    """Analiza un nivel de concurrencia (`workers` ∈ 1..8) desde un estado real de campaña."""
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError(f"workers debe ser entero en [1, 8]; recibido {workers!r}")
    state_dir = Path(state_dir)
    if state_dir.is_symlink() or not state_dir.is_dir():
        raise ValueError(f"Directorio de estado inexistente o enlace simbólico: {state_dir}")
    queue_dir = state_dir / "queue"
    if queue_dir.is_symlink() or not queue_dir.is_dir():
        raise ValueError(f"Falta directorio queue/ en {state_dir}")

    job_rows: list[dict[str, Any]] = []
    for q_path in sorted(queue_dir.glob("*.json")):
        if q_path.is_symlink():
            raise ValueError(f"Enlace simbólico prohibido en cola: {q_path}")
        q_doc = json.loads(q_path.read_text(encoding="utf-8"))
        model = q_doc.get("model")
        year = q_doc.get("year")
        seed = q_doc.get("seed")
        fit_report = None
        if model is not None and year is not None and seed is not None:
            fit_path = (
                state_dir / "runs" / str(model) / str(year) / f"seed-{seed}" / "fit" / "run.json"
            )
            if fit_path.is_symlink():
                raise ValueError(f"Enlace simbólico prohibido en informe de ajuste: {fit_path}")
            if fit_path.is_file():
                fit_report = json.loads(fit_path.read_text(encoding="utf-8"))
        job_rows.append(
            extract_job_telemetry(q_doc, fit_report, job_key=q_path.stem.replace("--", "/"))
        )

    if not job_rows:
        raise ValueError(f"No hay registros de trabajos en {queue_dir}")

    completed = [j for j in job_rows if j["status"] == "completed"]
    oom_count = sum(1 for j in job_rows if j["oom_detected"])
    failed_count = sum(1 for j in job_rows if j["status"] == "failed")
    rates = [j["steps_per_min"] for j in job_rows if j["steps"] > 0 and j["wall_seconds"] > 0]
    per_worker_median = statistics.median(rates) if rates else 0.0
    per_worker_mean = statistics.fmean(rates) if rates else 0.0
    effective_parallel = min(workers, len(rates)) if rates else 0
    aggregate_steps_per_min = per_worker_mean * effective_parallel

    rss_vals = [j["peak_rss_gib"] for j in job_rows if j["peak_rss_gib"] is not None]
    vram_res_vals = [j["vram_reserved_gib"] for j in job_rows if j["vram_reserved_gib"] is not None]
    vram_alloc_vals = [
        j["vram_allocated_gib"] for j in job_rows if j["vram_allocated_gib"] is not None
    ]

    max_job_rss_gib = max(rss_vals) if rss_vals else 0.0
    sum_job_rss_gib = sum(sorted(rss_vals, reverse=True)[:workers]) if rss_vals else 0.0
    max_job_vram_reserved_gib = max(vram_res_vals) if vram_res_vals else 0.0
    max_job_vram_allocated_gib = max(vram_alloc_vals) if vram_alloc_vals else 0.0

    samples = _read_samples_jsonl(samples_path)
    observed_host_ram_peak_gib = (
        samples["peak_host_ram_gib"]
        if samples["peak_host_ram_gib"] is not None
        else sum_job_rss_gib
    )
    est_vram_from_jobs = (
        sum(sorted(vram_res_vals, reverse=True)[:workers]) + workers * (CUDA_CONTEXT_MIB / 1024.0)
        if vram_res_vals
        else 0.0
    )
    observed_vram_peak_gib = (
        samples["peak_gpu_vram_gib"]
        if samples["peak_gpu_vram_gib"] is not None
        else est_vram_from_jobs
    )

    host_ram_total_gib = site_sheet.get("host_ram_total_gib") if site_sheet else None
    gpu_vram_total_gib = site_sheet.get("gpu_vram_total_gib") if site_sheet else None
    fp32_strict_ok = site_sheet.get("fp32_strict_ok", True) if site_sheet else True

    host_ram_pct = (
        (observed_host_ram_peak_gib / host_ram_total_gib) * 100.0 if host_ram_total_gib else None
    )
    host_ram_free_gib = (
        (host_ram_total_gib - observed_host_ram_peak_gib) if host_ram_total_gib else None
    )
    vram_pct = (observed_vram_peak_gib / gpu_vram_total_gib) * 100.0 if gpu_vram_total_gib else None
    vram_free_gib = (gpu_vram_total_gib - observed_vram_peak_gib) if gpu_vram_total_gib else None

    return {
        "workers": workers,
        "jobs_analyzed": len(job_rows),
        "completed_jobs": len(completed),
        "failed_count": failed_count,
        "oom_count": oom_count,
        "steps_total": sum(j["steps"] for j in job_rows),
        "per_worker_steps_per_min_median": per_worker_median,
        "per_worker_steps_per_min_mean": per_worker_mean,
        "aggregate_steps_per_min": aggregate_steps_per_min,
        "max_job_rss_gib": max_job_rss_gib,
        "sum_job_rss_gib": sum_job_rss_gib,
        "observed_host_ram_peak_gib": observed_host_ram_peak_gib,
        "host_ram_total_gib": host_ram_total_gib,
        "host_ram_pct": host_ram_pct,
        "host_ram_free_gib": host_ram_free_gib,
        "max_job_vram_allocated_gib": max_job_vram_allocated_gib,
        "max_job_vram_reserved_gib": max_job_vram_reserved_gib,
        "observed_vram_peak_gib": observed_vram_peak_gib,
        "gpu_vram_total_gib": gpu_vram_total_gib,
        "vram_pct": vram_pct,
        "vram_free_gib": vram_free_gib,
        "mean_gpu_util_pct": samples["mean_gpu_util_pct"],
        "samples_count": samples["count"],
        "fp32_strict_ok": fp32_strict_ok,
        "jobs": job_rows,
    }


def recommend_concurrency(
    level_reports: list[dict[str, Any]],
    *,
    max_ram_pct: float = 85.0,
    max_vram_pct: float = 85.0,
    min_ram_free_gib: float = 12.0,
    min_vram_free_gib: float = 6.0,
    min_marginal_gain_pct: float = 10.0,
) -> dict[str, Any]:
    """Selecciona el mayor nivel seguro con ganancia marginal real de pasos/min."""
    if not level_reports:
        raise ValueError("Se necesita al menos un informe de nivel para recomendar concurrencia")
    ordered = sorted(level_reports, key=lambda r: int(r["workers"]))
    baseline_rate = next(
        (
            float(r["aggregate_steps_per_min"])
            for r in ordered
            if r["workers"] == 1 and r["aggregate_steps_per_min"] > 0
        ),
        float(ordered[0]["aggregate_steps_per_min"]),
    )

    evaluated: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None

    for row in ordered:
        reasons: list[str] = []
        if row.get("oom_count", 0) > 0:
            reasons.append("oom_detected")
        if row.get("failed_count", 0) > 0 and row.get("oom_count", 0) == 0:
            reasons.append("job_failures")
        if not row.get("fp32_strict_ok", True):
            reasons.append("fp32_strict_violation")
        if row.get("host_ram_pct") is not None and row["host_ram_pct"] > max_ram_pct:
            reasons.append("host_ram_exceeded")
        if row.get("host_ram_free_gib") is not None and row["host_ram_free_gib"] < min_ram_free_gib:
            reasons.append("host_ram_margin_low")
        if row.get("vram_pct") is not None and row["vram_pct"] > max_vram_pct:
            reasons.append("vram_exceeded")
        if row.get("vram_free_gib") is not None and row["vram_free_gib"] < min_vram_free_gib:
            reasons.append("vram_margin_low")
        if row.get("aggregate_steps_per_min", 0.0) <= 0.0:
            reasons.append("zero_throughput")

        safe = len(reasons) == 0
        agg = float(row["aggregate_steps_per_min"])
        speedup_vs_1 = (agg / baseline_rate) if baseline_rate > 0.0 else 0.0
        efficiency_per_worker = speedup_vs_1 / float(row["workers"]) if row["workers"] > 0 else 0.0

        marginal_gain_pct: float | None = None
        if chosen is not None and chosen["aggregate_steps_per_min"] > 0:
            marginal_gain_pct = (
                (agg - chosen["aggregate_steps_per_min"]) / chosen["aggregate_steps_per_min"]
            ) * 100.0

        entry = dict(
            row,
            safe=safe,
            unsafe_reasons=reasons,
            speedup_vs_1=speedup_vs_1,
            efficiency_per_worker=efficiency_per_worker,
            marginal_gain_pct=marginal_gain_pct,
        )
        evaluated.append(entry)

        if safe:
            if chosen is None:
                chosen = entry
            elif marginal_gain_pct is not None and marginal_gain_pct >= min_marginal_gain_pct:
                chosen = entry

    return {
        "schema_version": 1,
        "kind": "colab_concurrency_probe_report_v1",
        "status": "ok" if chosen is not None else "unsafe",
        "recommended_workers": chosen["workers"] if chosen is not None else None,
        "recommended_aggregate_steps_per_min": chosen["aggregate_steps_per_min"]
        if chosen is not None
        else 0.0,
        "thresholds": {
            "max_ram_pct": max_ram_pct,
            "max_vram_pct": max_vram_pct,
            "min_ram_free_gib": min_ram_free_gib,
            "min_vram_free_gib": min_vram_free_gib,
            "min_marginal_gain_pct": min_marginal_gain_pct,
        },
        "levels": evaluated,
    }


def build_colab_probe_commands(
    *,
    kit_dir: Path,
    repo_dir: Path,
    state_root: Path,
    views_dir: Path,
    declaration: str,
    origin: int,
    levels: tuple[int, ...] = PROBE_LEVELS,
    site: str = "google-colab-rtx-pro-6000",
    seeds: str = "42 43 44",
    tiers: str = "core controls",
) -> list[dict[str, Any]]:
    """Genera las órdenes exactas de `colab_origin.sh` para la sonda de 1, 2, 4 y 6 trabajos."""
    if (
        isinstance(origin, bool)
        or not isinstance(origin, int)
        or origin < 2005
        or origin >= FORBIDDEN_YEAR
    ):
        raise ValueError(f"Origen inválido o sellado (2024 prohibido): {origin!r}")
    if _SITE_RE.fullmatch(site) is None:
        raise ValueError(f"MARS_TITAN_SITE inválido: {site!r}")
    assert_no_sealed_2024(
        {"declaration": declaration, "views": str(views_dir), "state": str(state_root)}
    )

    plan: list[dict[str, Any]] = []
    for workers in levels:
        if type(workers) is not int or not 1 <= workers <= 8:
            raise ValueError(f"Nivel de trabajadores inválido: {workers!r}")
        level_state = Path(state_root) / f"w{workers}"
        cmd = (
            f"bash {Path(kit_dir) / 'colab_origin.sh'} lanza "
            f"--repo {repo_dir} --estado {level_state} --vistas {views_dir} "
            f"--declaracion {declaration} --origen {origin} "
            f"--semillas '{seeds}' --niveles '{tiers}' --trabajadores {workers}"
        )
        plan.append(
            {
                "workers": workers,
                "state_dir": str(level_state),
                "samples_path": str(level_state / "samples.jsonl"),
                "command": cmd,
                "env": {
                    "MARS_TITAN_SITE": site,
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                    "TRITON_F32_DEFAULT": "ieee",
                    "PYTHONUNBUFFERED": "1",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True,max_split_size_mb:256",
                },
            }
        )
    return plan


def sample_host_and_gpu(
    *,
    meminfo_path: Path = Path("/proc/meminfo"),
    run=subprocess.run,
) -> dict[str, Any]:
    """Toma una muestra puntual de RAM del anfitrión y VRAM de GPU 0 para `samples.jsonl`."""
    mem = parse_meminfo(meminfo_path.read_text(encoding="utf-8"))
    gpu_used = None
    gpu_total = None
    gpu_util = None
    try:
        proc = run(
            [
                "nvidia-smi",
                "--id=0",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
        parsed_gpu = parse_nvidia_smi_memory(proc.stdout)
        gpu_used = parsed_gpu["used_mib"]
        gpu_total = parsed_gpu["total_mib"]
        gpu_util = parsed_gpu["util_pct"]
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "host_ram_total_gib": round(mem["MemTotal"] / (1024.0 * 1024.0), 3),
        "host_ram_used_gib": round(mem["used_kib"] / (1024.0 * 1024.0), 3),
        "gpu_mem_used_mib": gpu_used,
        "gpu_mem_total_mib": gpu_total,
        "gpu_util_pct": gpu_util,
    }


def run_live_probe_level(
    entry: dict[str, Any],
    *,
    duration_seconds: float = 300.0,
    sample_interval_seconds: float = 5.0,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Ejecuta en Colab un nivel de la sonda con muestreo de RAM/VRAM y parada ordenada SIGTERM."""
    if dry_run:
        return {"dry_run": True, "workers": entry["workers"], "command": entry["command"]}
    state_dir = Path(entry["state_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    samples_path = Path(entry["samples_path"])
    env = {**os.environ, **entry["env"]}
    proc = subprocess.Popen(["bash", "-lc", entry["command"]], env=env, start_new_session=True)
    deadline = time.monotonic() + duration_seconds
    samples_written = 0
    try:
        with samples_path.open("a", encoding="utf-8") as stream:
            while proc.poll() is None:
                sample = sample_host_and_gpu()
                stream.write(json.dumps(sample) + "\n")
                stream.flush()
                samples_written += 1
                if time.monotonic() >= deadline:
                    os.killpg(proc.pid, signal.SIGTERM)
                    break
                time.sleep(sample_interval_seconds)
        code = proc.wait(timeout=180)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
    return {
        "dry_run": False,
        "workers": entry["workers"],
        "exit_code": code,
        "samples_written": samples_written,
        "samples_path": str(samples_path),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Sonda de concurrencia (1, 2, 4, 6) para Colab RTX PRO 6000."
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    p_plan = sub.add_parser(
        "plan", help="Imprime plan y órdenes exactas para 1, 2, 4 y 6 trabajadores."
    )
    p_plan.add_argument("--kit", type=Path, required=True)
    p_plan.add_argument("--repo", type=Path, required=True)
    p_plan.add_argument("--state-root", type=Path, required=True)
    p_plan.add_argument("--views", type=Path, required=True)
    p_plan.add_argument("--declaration", required=True)
    p_plan.add_argument("--origin", type=int, default=2013)
    p_plan.add_argument("--site", default="google-colab-rtx-pro-6000")
    p_plan.add_argument("--seeds", default="42 43 44")
    p_plan.add_argument("--tiers", default="core controls")
    p_plan.add_argument(
        "--run-live", action="store_true", help="Ejecuta los niveles en Colab con muestreo."
    )
    p_plan.add_argument("--duration-seconds", type=float, default=300.0)

    p_an = sub.add_parser("analyze", help="Analiza estados w1/w2/w4/w6 generados por la sonda.")
    p_an.add_argument("--probe-root", type=Path, required=True)
    p_an.add_argument("--site-json", type=Path, default=None)
    p_an.add_argument("--levels", default="1,2,4,6")
    p_an.add_argument("--output", type=Path, default=None)

    args = parser.parse_args(argv)
    if args.mode == "plan":
        levels = build_colab_probe_commands(
            kit_dir=args.kit,
            repo_dir=args.repo,
            state_root=args.state_root,
            views_dir=args.views,
            declaration=args.declaration,
            origin=args.origin,
            site=args.site,
            seeds=args.seeds,
            tiers=args.tiers,
        )
        runs = []
        if args.run_live:
            for entry in levels:
                runs.append(run_live_probe_level(entry, duration_seconds=args.duration_seconds))
        doc = {
            "schema_version": 1,
            "kind": "colab_concurrency_probe_plan_v1",
            "levels": levels,
            "runs": runs,
        }
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        return 0

    site_info = load_site_sheet(args.site_json) if args.site_json else None
    parsed_levels = [int(x.strip()) for x in args.levels.split(",") if x.strip()]
    reports = []
    for w in parsed_levels:
        level_dir = args.probe_root / f"w{w}"
        if level_dir.is_dir():
            samples = level_dir / "samples.jsonl"
            reports.append(
                analyze_level_state(
                    level_dir,
                    workers=w,
                    site_sheet=site_info,
                    samples_path=samples if samples.is_file() else None,
                )
            )
    rec = recommend_concurrency(reports)
    text = json.dumps(rec, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if rec["status"] == "ok" else 2


if __name__ == "__main__":
    raise SystemExit(main())
