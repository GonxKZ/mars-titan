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
INVALID_REASONS = {None, "none", "missing_close", "unpriced_exit", "ruined", "incomplete"}
ADAPTATION_METHODS = {"reinforce", "expected", "mae", "klpo_full", "klpo_mc", "klpo_exact"}
NEURAL_CONTROLS = {"neural_mae", "neural_mse"}
CONDITIONS = {"real", "real_resampled", "real_synthetic"}
ADAPTIVE_VARIANTS = {
    "ppo",
    "double_dqn",
    "ppo_window",
    "ppo_gru",
    "ppo_episodic",
    "ppo_hmm",
    "ppo_episodic_hmm",
    "ppo_recent_aux",
    "ppo_replay_aux",
}


def native_adaptation(report):
    """Reconocer el contrato técnico explícito, sin ampliar otras versiones de productor."""
    audit = report.get("kind") == "native_ppo_audit"
    return (
        report.get("schema_version") in ({2} if audit else {2, 3})
        and report.get("kind") in {"native_ppo", "native_ppo_audit"}
        and report.get("activity") == ("evaluation" if audit else "rl")
        and report.get("model") in ADAPTIVE_VARIANTS
        and report.get("backend") == "native_libtorch"
        and report.get("domain") in {"synthetic", "technical"}
        and report.get("analysis_domain") == "technical"
        and report.get("final_test_opened") is False
        and report.get("macro_coverage") == dict(simulated_concepts=3, catalog_concepts=140)
        and (
            not audit or report.get("phase") == "evaluation" and report.get("partition") == "audit"
        )
    )


def classify(report, task, case):
    """Los informes anteriores conservan la actividad que acredita su coordinador."""
    if (
        report.get("model")
        in ADAPTIVE_VARIANTS
        | {
            "factor_world",
            "simulator",
            "cash",
            "hold_initial",
            "rebalance_50",
            "financial_comparison",
            "adaptive_comparison",
        }
        and "activity" not in report
    ):
        raise ValueError("El productor necesita declarar su actividad")
    if "activity" in report and report.get("schema_version") != 1 and not native_adaptation(report):
        raise ValueError("Versión del productor no admitida")
    mode = case.get("mode")
    inferred = (
        "predictive_adaptation"
        if mode
        else "supervised_continuation"
        if task.get("stage") == "posttraining"
        else "initial_training"
    )
    activity = report.get("activity", task.get("activity", inferred))
    valid_mode = mode in ADAPTATION_METHODS or (
        mode in NEURAL_CONTROLS and activity == "supervised_continuation"
    )
    if activity not in ACTIVITIES or mode and not valid_mode:
        raise ValueError("Actividad o método sin contrato público")
    return activity, mode if activity in PREDICTIVE and mode else activity


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
