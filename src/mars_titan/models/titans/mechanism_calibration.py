"""Calibración por cohorte y retornos que conservan el contexto ya elegido antes de MAC."""

import numpy as np

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.calibration.cohort_feedback import CohortCapacityError

from .config import canonical


def _static_record(payload):
    cal = payload["calibrator"]
    return dict(
        schema_version=1,
        kind=cqr.KIND,
        method=cqr.METHOD,
        order_rule=cqr.ORDER_RULE,
        coverage_guaranteed=False,
        levels=cal["header"]["levels"],
        nominals=cal["header"]["nominals"],
        groups={
            name: dict(
                intervals={
                    key: dict(correction=value["correction"])
                    for key, value in market["intervals"].items()
                }
            )
            for name, market in cal["markets"].items()
        },
    )


def _reserve_fits(calibration):
    payload = calibration.export()
    unresolved = sum(
        row["available_at"] is None for item in payload["cohorts"] for row in item["rows"]
    )
    # Reserva operativa para disponibilidad, estado y evidencia de resultados parciales.
    # No es una cota de RAM ni una certificación de aritmética o cobertura.
    return len(canonical(payload).encode()) + 128 * unresolved + 1024 <= payload["max_bytes"]


def calibrate_event(event, rows, static):
    calibration = event.calibration
    online = [group for group, plan in event.plans if plan.applied.calibration == "online"]
    proposed, online_values = calibration, {}
    work = dict(
        online_attempts=0,
        online_failures=0,
        online_created_proposals=0,
        online_selected_proposals=0,
        online_discarded_proposals=0,
        static_cohorts=0,
    )
    capacity_failure = False
    for group in online:
        indices = [i for i, flow in enumerate(event.flow_ids) if flow.startswith(group + "/")]
        try:
            work["online_attempts"] += 1
            emission = proposed.emit(
                group,
                event.cutoff,
                tuple(rows[i]["sample_id"] for i in indices),
                np.array([rows[i]["raw_quantiles"] for i in indices]),
                tuple(event.matures_at[i] for i in indices),
            )
            work["online_created_proposals"] += 1
            if not _reserve_fits(emission.state):
                raise CohortCapacityError("PT2 no conserva reserva para resultados parciales")
        except CohortCapacityError:
            work["online_failures"] += 1
            capacity_failure = True
            break
        proposed = emission.state
        online_values[group] = emission.quantiles
    if capacity_failure:
        work["online_discarded_proposals"] = work["online_created_proposals"]
        for row in rows:
            if row["calibration"] == "online":
                row.update(
                    point=row["k1_point"],
                    raw_quantiles=row["k1_quantiles"],
                    emitted_route="k1",
                    calibration="none",
                    reasons=(*row["reasons"], "pending_calibration_bytes"),
                    discarded_reader_steps=max(0, row["reader_steps"] - 1),
                )
    else:
        work["online_selected_proposals"] = work["online_created_proposals"]
        calibration = proposed
    raw = (
        None
        if rows[0]["raw_quantiles"] is None
        else np.array([row["raw_quantiles"] for row in rows])
    )
    output = None if raw is None else raw.copy()
    for group, _plan in event.plans:
        indices = [i for i, flow in enumerate(event.flow_ids) if flow.startswith(group + "/")]
        mode = rows[indices[0]]["calibration"]
        if mode == "static":
            work["static_cohorts"] += 1
            output[indices], _ = cqr.apply_conformal_quantiles(
                static, raw[indices], np.array([group] * len(indices))
            )
        elif mode == "online":
            output[indices] = online_values[group]
    if output is not None and not np.isfinite(output).all():
        raise ValueError("La calibración produjo cuantiles no finitos")
    return calibration, raw, output, work
