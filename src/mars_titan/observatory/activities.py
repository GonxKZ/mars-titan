"""Actividades y medidas públicas de experimentos con objetivos distintos."""

import math

ACTIVITIES = {
    "initial_training",
    "supervised_continuation",
    "predictive_adaptation",
    "rl",
    "synthetic_generation",
    "simulation",
    "evaluation",
}
PREDICTIVE = {"initial_training", "supervised_continuation", "predictive_adaptation"}
FINANCIAL = {"rl", "simulation", "evaluation"}
INVALID_REASONS = {None, "none", "missing_close", "ruined", "incomplete"}
ADAPTATION_METHODS = {"reinforce", "expected", "mae", "klpo_full", "klpo_mc", "klpo_exact"}


def classify(report, task, case):
    """Los informes anteriores conservan la actividad que acredita su coordinador."""
    if (
        report.get("model")
        in {
            "factor_world",
            "ppo",
            "double_dqn",
            "simulator",
            "cash",
            "hold_initial",
            "rebalance_50",
            "financial_comparison",
        }
        and "activity" not in report
    ):
        raise ValueError("El productor necesita declarar su actividad")
    if "activity" in report and report.get("schema_version") != 1:
        raise ValueError("Versión del productor no admitida")
    mode = case.get("mode")
    inferred = (
        "predictive_adaptation"
        if mode
        else "supervised_continuation"
        if task.get("stage") == "posttraining"
        else "initial_training"
    )
    activity = report.get("activity", inferred)
    if activity not in ACTIVITIES or mode and mode not in ADAPTATION_METHODS:
        raise ValueError("Actividad o método sin contrato público")
    return activity, mode if activity == "predictive_adaptation" and mode else activity


def financial_validation(report, activity):
    """Exportar solo agregados de validación de un motor sin deuda ni cortos."""
    if (
        activity not in FINANCIAL
        or report.get("final_test_opened") is not False
        or report.get("phase") in {"test", "evaluation"}
        or report.get("partition", "validation") != "validation"
    ):
        return None
    raw = report.get("financial_validation")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("La validación financiera necesita un objeto")
    if raw.get("partition", "validation") != "validation":
        return None
    values = {
        key: raw.get(key)
        for key in (
            "net_return",
            "max_drawdown",
            "costs",
            "turnover",
            "steps",
            "completed",
            "invalid_reason",
        )
    }
    for key in ("net_return", "max_drawdown", "costs", "turnover", "steps"):
        value = values[key]
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("El agregado financiero debe ser numérico y finito")
        low = -1 if key == "net_return" else 0
        high = 1 if key == "max_drawdown" else math.inf
        if not low <= value <= high:
            raise ValueError("El agregado financiero está fuera de rango")
        if key == "steps" and (type(value) is not int or value > 2**53 - 1):
            raise ValueError("Los pasos financieros necesitan un entero acotado")
    if values["completed"] is not None and type(values["completed"]) is not bool:
        raise ValueError("El cierre financiero necesita un booleano")
    if values["invalid_reason"] not in INVALID_REASONS:
        raise ValueError("El motivo financiero no pertenece al vocabulario público")
    return values
