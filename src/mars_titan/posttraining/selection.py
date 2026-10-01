"""Política versionada de selección para ajustes con presupuesto fijo o paciencia."""

from mars_titan.training.selection import advance_selection, initial_selection, validate_selection


def selection_policy(case):
    requested = case.get("selection")
    if "selection" not in case:
        return dict(version=1, metric="session_mae", patience=None, min_delta=0.0)
    keys = {"version", "metric", "patience", "min_delta"}
    if isinstance(requested, dict) and requested.get("version") == 3:
        keys.add("minimum_epochs")
    if (
        not isinstance(requested, dict)
        or set(requested) != keys
        or type(requested["version"]) is not int
        or requested["version"] not in {2, 3}
    ):
        raise ValueError("La política de selección debe declarar la versión 2 o 3")
    if requested["version"] == 3 and (requested["patience"] is None or case["condition"] != "real"):
        raise ValueError("La selección con mínimo requiere paciencia y continuación real")
    if requested["patience"] is not None and case["condition"] != "real":
        raise ValueError("La parada por paciencia solo admite continuaciones reales separadas")
    validate_selection(_options(requested, case["epochs"]), epochs=case["epochs"])
    return dict(requested)


def _options(policy, epochs):
    options = dict(
        metric=policy["metric"],
        patience=epochs if policy["patience"] is None else policy["patience"],
        min_delta=policy["min_delta"],
    )
    if policy["version"] == 3:
        options["minimum_epochs"] = policy["minimum_epochs"]
    return options


def select_epoch(previous, score, epoch, policy, epochs):
    initial = type(epoch) is int and epoch == 0
    if initial and previous is not None:
        raise ValueError("La validación inicial ya está confirmada")
    if initial:
        return initial_selection(score, _options(policy, epochs))
    return advance_selection(previous, score, epoch, _options(policy, epochs))
