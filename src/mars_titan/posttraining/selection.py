"""Política versionada de selección para ajustes con presupuesto fijo o paciencia."""

from mars_titan.training.selection import advance_selection, validate_selection


def selection_policy(case):
    requested = case.get("selection")
    if "selection" not in case:
        return dict(version=1, metric="session_mae", patience=None, min_delta=0.0)
    if (
        not isinstance(requested, dict)
        or set(requested) != {"version", "metric", "patience", "min_delta"}
        or type(requested["version"]) is not int
        or requested["version"] != 2
    ):
        raise ValueError("La política de selección debe declarar la versión 2")
    if requested["patience"] is not None and case["condition"] != "real":
        raise ValueError("La parada por paciencia solo admite continuaciones reales separadas")
    validate_selection(_options(requested, case["epochs"]))
    return dict(requested)


def _options(policy, epochs):
    return dict(
        metric=policy["metric"],
        patience=epochs if policy["patience"] is None else policy["patience"],
        min_delta=policy["min_delta"],
    )


def select_epoch(previous, score, epoch, policy, epochs):
    initial = type(epoch) is int and epoch == 0
    if initial and previous is not None:
        raise ValueError("La validación inicial ya está confirmada")
    result = advance_selection(previous, score, 1 if initial else epoch, _options(policy, epochs))
    if initial:
        # La evaluación inicial no consume una época de ajuste ni paciencia.
        result.update(last_epoch=0, best_epoch=0)
    return result
