#!/usr/bin/env python3
"""Calculadora de unidades de cómputo (CU) y política automática de parada/sincronización.

Integra la tarifa de CU/hora leída del panel RAM/Disk de Google Colab, un libro contable
persistente entre relevos de 24 h (`cu_ledger.json`), el presupuesto y margen de reserva,
el reloj de sesión (corte preventivo a 22,5 h) y la tasa real de `pasos/min` medida en la
campaña para decidir si lanzar el siguiente origen/trabajo o parar limpiamente guardando
estado y ejecutando `colab_sync.py push` hacia Drive.

Excluye estrictamente el año sellado 2024 y etiqueta toda cifra no medida en silicio real
como `[no verificado en GPU]`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

FORBIDDEN_YEAR = 2024
FIRST_YEAR = 2005
DEFAULT_BUDGET_CU = 1000.0
DEFAULT_SESSION_HARD_HOURS = 24.0
DEFAULT_SESSION_SOFT_HOURS = 22.5
DEFAULT_RESERVE_FRACTION = 0.15
UNVERIFIED_GPU_TAG = "[no verificado en GPU]"
BASE_DGX_HOURS_2013 = (
    1.6  # Base uniforme coherente en todos los bloques (entre 1,2 h mediana y 2,1 h titans_mac)
)

EXIT_LAUNCH = 0
EXIT_STOP_CU_RESERVE = 10
EXIT_STOP_SESSION_WALL = 11
EXIT_STOP_JOB_EXCEEDS_BUDGET = 12
EXIT_ERROR = 2

# [Medición del proyecto] Filas de ajuste reales en dgx-us-campaign.md:110-116,584-586
_MEASURED_US_ROWS: tuple[tuple[int, float], ...] = (
    (2005, 1_313_128.0),
    (2010, 3_602_637.0),
    (2013, 5_353_824.0),
    (2015, 6_667_897.0),
    (2020, 10_722_076.0),
    (2023, 13_775_654.0),
)
_ROWS_2013_US = 5_353_824.0
_ROWS_2013_US_CN = 6_080_386.0
_ROWS_2023_US_CN = 16_204_399.0

_RATE_PATTERNS = (
    re.compile(
        r"(-?\d+(?:[.,]\d+)?)\s*(?:compute\s+units?\s*(?:per|/)\s*(?:hour|hr|h)|cu\s*/\s*h|unidades(?:\s+de\s+c[oó]mputo)?\s*(?:por|/)\s*hora|units?\s*/\s*(?:hour|hr))",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:rate|tarifa|tasa)\s*[:=]\s*(-?\d+(?:[.,]\d+)?)\s*(?:/\s*(?:hr|h|hora))?", re.IGNORECASE
    ),
)
_AVAIL_PATTERNS = (
    re.compile(
        r"(?:available|disponibles?|remaining|saldo)\s*[:=]?\s*(-?\d+(?:[.,]\d+)?)", re.IGNORECASE
    ),
)


def _to_float(token: str) -> float:
    value = float(token.replace(",", "."))
    if not math.isfinite(value):
        raise ValueError(f"Valor numérico no finito: {token!r}")
    return value


def parse_colab_cu_panel(text: str) -> dict[str, Any]:
    """Lee la tarifa de CU/hora (y saldo disponible opcional) pegada del panel de Colab o JSON."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("El texto del panel de Colab está vacío")
    stripped = text.strip()
    if stripped.startswith("{"):
        try:
            doc = json.loads(stripped)
        except ValueError as error:
            raise ValueError(f"JSON del panel inválido: {error}") from error
        rate = doc.get("cu_per_hour")
        avail = doc.get("available_cu")
        if (
            isinstance(rate, bool)
            or not isinstance(rate, (int, float))
            or not math.isfinite(rate)
            or rate <= 0
        ):
            raise ValueError(f"cu_per_hour inválido en JSON: {rate!r}")
        if avail is not None:
            if (
                isinstance(avail, bool)
                or not isinstance(avail, (int, float))
                or not math.isfinite(avail)
                or avail < 0
            ):
                raise ValueError(f"available_cu inválido en JSON: {avail!r}")
            avail = float(avail)
        return {"cu_per_hour": float(rate), "available_cu": avail, "source": "json"}

    rate_val: float | None = None
    for pat in _RATE_PATTERNS:
        m = pat.search(stripped)
        if m is not None:
            rate_val = _to_float(m.group(1))
            break
    if rate_val is None or rate_val <= 0.0:
        raise ValueError(f"No se pudo extraer una tarifa CU/h positiva de: {text!r}")

    avail_val: float | None = None
    for pat in _AVAIL_PATTERNS:
        m = pat.search(stripped)
        if m is not None:
            avail_val = _to_float(m.group(1))
            if avail_val < 0.0:
                raise ValueError(f"Saldo de CU negativo en panel: {avail_val}")
            break
    return {"cu_per_hour": rate_val, "available_cu": avail_val, "source": "panel_text"}


def _interp_us_rows(year: int) -> float:
    for idx in range(len(_MEASURED_US_ROWS) - 1):
        y0, r0 = _MEASURED_US_ROWS[idx]
        y1, r1 = _MEASURED_US_ROWS[idx + 1]
        if y0 <= year <= y1:
            frac = (year - y0) / float(y1 - y0)
            return r0 + frac * (r1 - r0)
    raise ValueError(f"Año fuera de rango medido: {year}")


def origin_row_multiplier(year: int, market: str = "US") -> float:
    """Factor de filas respecto a EE. UU. 2013 (rechaza cualquier origen >= 2024)."""
    if isinstance(year, bool) or not isinstance(year, int):
        raise ValueError(f"Año inválido: {year!r}")
    if year >= FORBIDDEN_YEAR:
        raise ValueError(f"Origen {year} prohibido: 2024 permanece sellado")
    if year < FIRST_YEAR:
        raise ValueError(f"Origen {year} anterior a {FIRST_YEAR}")
    if market not in ("US", "US+CN"):
        raise ValueError(f"Mercado no soportado: {market!r}")
    us_rows = _interp_us_rows(year)
    if market == "US":
        return us_rows / _ROWS_2013_US
    r2013 = _ROWS_2013_US_CN / _ROWS_2013_US
    r2023 = _ROWS_2023_US_CN / 13_775_654.0
    frac = max(0.0, min(1.0, (year - 2013) / 10.0))
    return (us_rows * (r2013 + frac * (r2023 - r2013))) / _ROWS_2013_US


def estimate_remaining_job_wall_hours(
    *,
    remaining_steps: int,
    steps_per_min_aggregate: float,
    relay_overhead_fraction: float = 0.08,
) -> float:
    """Estima horas de pared necesarias a partir de la tasa real agregada de pasos/min."""
    if (
        isinstance(remaining_steps, bool)
        or not isinstance(remaining_steps, int)
        or remaining_steps < 0
    ):
        raise ValueError(f"remaining_steps debe ser entero >= 0; recibido {remaining_steps!r}")
    if (
        isinstance(steps_per_min_aggregate, bool)
        or not isinstance(steps_per_min_aggregate, (int, float))
        or not math.isfinite(steps_per_min_aggregate)
        or steps_per_min_aggregate <= 0.0
    ):
        raise ValueError(
            f"steps_per_min_aggregate debe ser > 0; recibido {steps_per_min_aggregate!r}"
        )
    if not (0.0 <= relay_overhead_fraction < 1.0):
        raise ValueError(
            f"relay_overhead_fraction debe estar en [0, 1); recibido {relay_overhead_fraction!r}"
        )
    if remaining_steps == 0:
        return 0.0
    base_hours = (remaining_steps / float(steps_per_min_aggregate)) / 60.0
    return base_hours * (1.0 + relay_overhead_fraction)


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    data = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    try:
        with tmp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _require_finite_number(
    value: Any,
    name: str,
    *,
    strictly_positive: bool = False,
    non_negative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} debe ser un número finito válido; recibido {value!r}")
    fval = float(value)
    if strictly_positive and fval <= 0.0:
        raise ValueError(f"{name} debe ser > 0; recibido {fval}")
    if non_negative and fval < 0.0:
        raise ValueError(f"{name} no puede ser negativo; recibido {fval}")
    return fval


def load_and_update_cu_ledger(
    ledger_path: Path,
    *,
    session_id: str,
    session_elapsed_hours: float,
    cu_per_hour: float,
    total_budget_cu: float = DEFAULT_BUDGET_CU,
) -> dict[str, Any]:
    """Libro acumulado de CU entre relevos de 24 h (nunca pierde CUs de sesiones anteriores)."""
    ledger_path = Path(ledger_path)
    if ledger_path.is_symlink():
        raise ValueError(f"El libro de CU no puede ser un enlace simbólico: {ledger_path}")
    if not session_id or not isinstance(session_id, str):
        raise ValueError("session_id debe ser una cadena no vacía")
    session_elapsed_hours = _require_finite_number(
        session_elapsed_hours, "session_elapsed_hours", non_negative=True
    )
    cu_per_hour = _require_finite_number(cu_per_hour, "cu_per_hour", strictly_positive=True)
    total_budget_cu = _require_finite_number(
        total_budget_cu, "total_budget_cu", strictly_positive=True
    )

    sessions: dict[str, dict[str, Any]] = {}
    if ledger_path.is_file():
        doc = json.loads(ledger_path.read_text(encoding="utf-8"))
        if doc.get("kind") != "colab_cu_ledger_v1" or not isinstance(doc.get("sessions"), dict):
            raise ValueError("Esquema inválido en cu_ledger.json")
        sessions = dict(doc["sessions"])

    prior_cu = sum(float(entry["cu_spent"]) for sid, entry in sessions.items() if sid != session_id)
    current_cu = float(session_elapsed_hours) * float(cu_per_hour)
    sessions[session_id] = {
        "elapsed_hours": round(float(session_elapsed_hours), 4),
        "cu_per_hour": round(float(cu_per_hour), 4),
        "cu_spent": round(current_cu, 4),
        "updated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    total_cu_spent = prior_cu + current_cu
    payload = {
        "schema_version": 1,
        "kind": "colab_cu_ledger_v1",
        "total_budget_cu": float(total_budget_cu),
        "prior_relays_cu_spent": round(prior_cu, 4),
        "current_session_id": session_id,
        "current_session_cu_spent": round(current_cu, 4),
        "total_cu_spent": round(total_cu_spent, 4),
        "remaining_cu": round(float(total_budget_cu) - total_cu_spent, 4),
        "sessions": sessions,
    }
    _atomic_write_json(ledger_path, payload)
    return payload


def _build_shutdown_protocol(
    state_dir: Path | None, mirror_dir: Path | None, kit_dir: Path | None
) -> list[str]:
    kit = str(kit_dir) if kit_dir else "$K"
    state = str(state_dir) if state_dir else "$T/estado/v11"
    mirror = str(mirror_dir) if mirror_dir else "$MIRROR_DRIVE"
    return [
        (
            "1. Enviar SIGTERM a la sesión tmux/cola para que comparison_campaign.py cierre cada "
            "trabajo en su barrera de checkpoint (código 3)."
        ),
        (
            f"2. Ejecutar sincronización coherente: python3 {kit}/colab_sync.py push --state "
            f"{state} --mirror {mirror}"
        ),
        (
            "3. Verificar en el JSON de salida de colab_sync.py que held == [] y retried_later == "
            "[]; si no, reintentar push."
        ),
        (
            f"4. Guardar/espejar cu_ledger.json en {mirror}/cu_ledger.json para conservar el gasto "
            "acumulado entre relevos."
        ),
        (
            "5. Desconectar la VM con google.colab.runtime.unassign() (solo tras confirmar el "
            "paso 3) para no quemar CU en vacío."
        ),
    ]


def evaluate_launch_or_stop(
    *,
    cu_per_hour: float,
    total_budget_cu: float,
    prior_relays_cu_spent: float,
    session_elapsed_hours: float,
    next_job_estimated_wall_hours: float,
    next_origin_year: int,
    reserve_cu: float | None = None,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
    session_soft_limit_hours: float = DEFAULT_SESSION_SOFT_HOURS,
    min_session_margin_hours: float = 0.5,
    state_dir: Path | None = None,
    mirror_dir: Path | None = None,
    kit_dir: Path | None = None,
) -> dict[str, Any]:
    """Decide automáticamente lanzar el siguiente origen/trabajo o parar y sincronizar."""
    # Valida sellado de 2024 y rango de año
    origin_row_multiplier(next_origin_year, "US")
    cu_per_hour = _require_finite_number(cu_per_hour, "cu_per_hour", strictly_positive=True)
    total_budget_cu = _require_finite_number(
        total_budget_cu, "total_budget_cu", strictly_positive=True
    )
    prior_relays_cu_spent = _require_finite_number(
        prior_relays_cu_spent, "prior_relays_cu_spent", non_negative=True
    )
    session_elapsed_hours = _require_finite_number(
        session_elapsed_hours, "session_elapsed_hours", non_negative=True
    )
    next_job_estimated_wall_hours = _require_finite_number(
        next_job_estimated_wall_hours, "next_job_estimated_wall_hours", non_negative=True
    )
    reserve_fraction = _require_finite_number(
        reserve_fraction, "reserve_fraction", non_negative=True
    )
    if reserve_fraction >= 1.0:
        raise ValueError("reserve_fraction debe estar en [0, 1)")
    session_soft_limit_hours = _require_finite_number(
        session_soft_limit_hours, "session_soft_limit_hours", strictly_positive=True
    )
    min_session_margin_hours = _require_finite_number(
        min_session_margin_hours, "min_session_margin_hours", non_negative=True
    )
    if reserve_cu is not None:
        reserve_cu = _require_finite_number(reserve_cu, "reserve_cu", non_negative=True)

    effective_reserve = (
        float(reserve_cu)
        if reserve_cu is not None
        else float(total_budget_cu) * float(reserve_fraction)
    )
    if effective_reserve < 0 or effective_reserve >= total_budget_cu:
        raise ValueError("La reserva de CU debe estar en [0, total_budget_cu)")

    current_session_cu = float(session_elapsed_hours) * float(cu_per_hour)
    total_cu_spent = float(prior_relays_cu_spent) + current_session_cu
    usable_cu_remaining = float(total_budget_cu) - effective_reserve - total_cu_spent
    next_job_cu = float(next_job_estimated_wall_hours) * float(cu_per_hour)
    session_hours_remaining = float(session_soft_limit_hours) - float(session_elapsed_hours)

    if usable_cu_remaining <= 0.0:
        action = "stop_and_sync"
        exit_code = EXIT_STOP_CU_RESERVE
        reason = "cu_reserve_reached"
    elif session_hours_remaining < min_session_margin_hours:
        action = "stop_and_sync"
        exit_code = EXIT_STOP_SESSION_WALL
        reason = "session_wall_limit_reached"
    elif next_job_cu > usable_cu_remaining:
        action = "stop_and_sync"
        exit_code = EXIT_STOP_JOB_EXCEEDS_BUDGET
        reason = "next_job_exceeds_usable_cu"
    else:
        action = "launch"
        exit_code = EXIT_LAUNCH
        reason = "within_cu_and_session_budget"

    fits_current_relay = next_job_estimated_wall_hours <= session_hours_remaining
    return {
        "schema_version": 1,
        "kind": "colab_cu_guard_decision_v1",
        "action": action,
        "exit_code": exit_code,
        "reason": reason,
        "next_origin_year": next_origin_year,
        "cu_per_hour": round(float(cu_per_hour), 4),
        "total_budget_cu": round(float(total_budget_cu), 4),
        "effective_reserve_cu": round(effective_reserve, 4),
        "prior_relays_cu_spent": round(float(prior_relays_cu_spent), 4),
        "current_session_cu_spent": round(current_session_cu, 4),
        "total_cu_spent": round(total_cu_spent, 4),
        "usable_cu_remaining": round(usable_cu_remaining, 4),
        "next_job_estimated_wall_hours": round(float(next_job_estimated_wall_hours), 4),
        "next_job_cu": round(next_job_cu, 4),
        "session_hours_remaining": round(session_hours_remaining, 4),
        "fits_current_relay": fits_current_relay,
        "relay_warning": None
        if fits_current_relay
        else (
            "El trabajo supera el resto de esta sesión de 22,5 h; se pausará en checkpoint y "
            "seguirá en el próximo relevo."
        ),
        "shutdown_protocol": _build_shutdown_protocol(state_dir, mirror_dir, kit_dir),
    }


def _block_hours(block: dict[str, Any]) -> tuple[int, float]:
    origins = block["origins"]
    market = block["market"]
    models = int(block["models"])
    seeds = int(block["seeds"])
    base_h = float(block["base_dgx_hours_2013"])
    mult = sum(origin_row_multiplier(int(y), market) for y in origins)
    return len(origins) * models * seeds, mult * models * seeds * base_h


def build_campaign_cu_scenarios(
    *,
    cu_per_hour: float = 7.8,
    workers: int = 4,
    speedup_vs_dgx: float = 1.5,
    relay_overhead_fraction: float = 0.08,
    budget_cu: float = DEFAULT_BUDGET_CU,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> dict[str, Any]:
    """Escenarios de campaña alineados con `repo-v11` (base homogénea 1,6 h DGX)."""
    if cu_per_hour <= 0 or speedup_vs_dgx <= 0 or not (1 <= workers <= 8):
        raise ValueError("Parámetros de escenario inválidos")
    usable_budget = budget_cu * (1.0 - reserve_fraction)
    specs = [
        {
            "name": "S1_parity_v10_and_v11_screen1_2013",
            "description": (
                "Paridad v10 (titans_mac × 3 semillas) + criba us-campaign-v11-screen1.json (20 "
                "modelos × 3 semillas en 2013)."
            ),
            "blocks": [
                {
                    "origins": [2013],
                    "market": "US",
                    "models": 1,
                    "seeds": 3,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
                {
                    "origins": [2013],
                    "market": "US",
                    "models": 20,
                    "seeds": 3,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
            ],
        },
        {
            "name": "S2_hpo_us_v11_search1_2013",
            "description": (
                "Búsqueda conjunta us-v11-search1 (8 startup + 3 tandas × 6 = 26 ensayos × 1 "
                "semilla + top-3 × 2 semillas extra)."
            ),
            "blocks": [
                {
                    "origins": [2013],
                    "market": "US",
                    "models": 26,
                    "seeds": 1,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
                {
                    "origins": [2013],
                    "market": "US",
                    "models": 3,
                    "seeds": 2,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
            ],
        },
        {
            "name": "S3_clean_2014_2023_us_4models",
            "description": (
                "Orígenes limpios 2014–2023 EE. UU.: núcleo elegido + frozen + disabled + 1 "
                "candidato ganador (4 modelos × 3 semillas = 120 trabajos)."
            ),
            "blocks": [
                {
                    "origins": list(range(2014, 2024)),
                    "market": "US",
                    "models": 4,
                    "seeds": 3,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
            ],
        },
        {
            "name": "S4_recommended_within_850cu_usable",
            "description": (
                "Plan recomendado respetando 15 % reserva (<= 850 CU): Paridad v10 + criba v11 (1 "
                "semilla inicial + confirmación top-6 × 2 semillas) + 10 orígenes limpios EE. UU. "
                "(3 modelos × 3 semillas)."
            ),
            "blocks": [
                {
                    "origins": [2013],
                    "market": "US",
                    "models": 1,
                    "seeds": 3,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
                {
                    "origins": [2013],
                    "market": "US",
                    "models": 20,
                    "seeds": 1,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
                {
                    "origins": [2013],
                    "market": "US",
                    "models": 6,
                    "seeds": 2,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
                {
                    "origins": list(range(2014, 2024)),
                    "market": "US",
                    "models": 3,
                    "seeds": 3,
                    "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                },
            ],
        },
    ]

    scenarios = []
    for sp in specs:
        jobs = 0
        dgx_h = 0.0
        for blk in sp["blocks"]:
            j, h = _block_hours(blk)
            jobs += j
            dgx_h += h
        wall_h = (dgx_h / (workers * speedup_vs_dgx)) * (1.0 + relay_overhead_fraction)
        est_cu = wall_h * cu_per_hour
        relays = max(1, int(math.ceil(wall_h / DEFAULT_SESSION_SOFT_HOURS)))
        scenarios.append(
            {
                "name": sp["name"],
                "description": sp["description"],
                "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
                "total_jobs": jobs,
                "dgx_worker_hours": round(dgx_h, 2),
                "colab_wall_hours": round(wall_h, 2),
                "relay_sessions_22_5h": relays,
                "estimated_cu": round(est_cu, 2),
                "fits_total_budget": est_cu <= budget_cu,
                "fits_usable_budget_after_reserve": est_cu <= usable_budget,
                "remaining_cu_after_reserve": round(usable_budget - est_cu, 2),
            }
        )

    return {
        "schema_version": 1,
        "kind": "colab_cu_scenarios_v1",
        "budget_cu": budget_cu,
        "reserve_fraction": reserve_fraction,
        "usable_budget_cu": usable_budget,
        "assumptions": {
            "cu_per_hour": cu_per_hour,
            "workers": workers,
            "speedup_vs_dgx": speedup_vs_dgx,
            "relay_overhead_fraction": relay_overhead_fraction,
            "base_dgx_hours_2013": BASE_DGX_HOURS_2013,
        },
        "rate_status": UNVERIFIED_GPU_TAG,
        "scenarios": scenarios,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Calculadora de CU y política automática de parada en Colab."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_dec = sub.add_parser(
        "decide", help="Decide si lanzar el siguiente origen/trabajo o parar y sincronizar."
    )
    p_dec.add_argument(
        "--panel-text", required=True, help="Tarifa leída del panel de Colab o JSON."
    )
    p_dec.add_argument("--budget-cu", type=float, default=DEFAULT_BUDGET_CU)
    p_dec.add_argument("--prior-cu-spent", type=float, default=0.0)
    p_dec.add_argument("--session-elapsed-hours", type=float, required=True)
    p_dec.add_argument("--remaining-steps", type=int, default=None)
    p_dec.add_argument("--steps-per-min", type=float, default=None)
    p_dec.add_argument("--next-job-hours", type=float, default=None)
    p_dec.add_argument("--next-origin", type=int, required=True)
    p_dec.add_argument("--reserve-fraction", type=float, default=DEFAULT_RESERVE_FRACTION)
    p_dec.add_argument("--ledger", type=Path, default=None)
    p_dec.add_argument("--session-id", default="relay-01")
    p_dec.add_argument("--state-dir", type=Path, default=None)
    p_dec.add_argument("--mirror-dir", type=Path, default=None)
    p_dec.add_argument("--kit-dir", type=Path, default=None)

    p_sc = sub.add_parser(
        "scenarios", help="Calcula escenarios presupuestarios alineados con repo-v11."
    )
    p_sc.add_argument("--cu-rate", type=float, default=7.8)
    p_sc.add_argument("--workers", type=int, default=4)
    p_sc.add_argument("--speedup", type=float, default=1.5)
    p_sc.add_argument("--budget-cu", type=float, default=DEFAULT_BUDGET_CU)

    args = parser.parse_args(argv)
    if args.cmd == "scenarios":
        doc = build_campaign_cu_scenarios(
            cu_per_hour=args.cu_rate,
            workers=args.workers,
            speedup_vs_dgx=args.speedup,
            budget_cu=args.budget_cu,
        )
        print(json.dumps(doc, ensure_ascii=False, indent=2))
        return 0

    try:
        panel = parse_colab_cu_panel(args.panel_text)
        prior_spent = args.prior_cu_spent
        if args.ledger is not None:
            ledger_doc = load_and_update_cu_ledger(
                args.ledger,
                session_id=args.session_id,
                session_elapsed_hours=args.session_elapsed_hours,
                cu_per_hour=panel["cu_per_hour"],
                total_budget_cu=args.budget_cu,
            )
            prior_spent = ledger_doc["prior_relays_cu_spent"]
        if args.next_job_hours is not None:
            next_hours = args.next_job_hours
        elif args.remaining_steps is not None and args.steps_per_min is not None:
            next_hours = estimate_remaining_job_wall_hours(
                remaining_steps=args.remaining_steps,
                steps_per_min_aggregate=args.steps_per_min,
            )
        else:
            raise ValueError("Indica --next-job-hours o (--remaining-steps y --steps-per-min)")
        decision = evaluate_launch_or_stop(
            cu_per_hour=panel["cu_per_hour"],
            total_budget_cu=args.budget_cu,
            prior_relays_cu_spent=prior_spent,
            session_elapsed_hours=args.session_elapsed_hours,
            next_job_estimated_wall_hours=next_hours,
            next_origin_year=args.next_origin,
            reserve_fraction=args.reserve_fraction,
            state_dir=args.state_dir,
            mirror_dir=args.mirror_dir,
            kit_dir=args.kit_dir,
        )
    except ValueError as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False))
        return EXIT_ERROR

    print(json.dumps(decision, ensure_ascii=False, indent=2))
    return int(decision["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
