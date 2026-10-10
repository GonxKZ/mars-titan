"""Simular reglas de parada sobre historiales de validación ya registrados.

Lee los `run.json` de ajustes de referencias confirmados, sin modificarlos, y aplica a cada
historial de `session_mae` la selección de `mars_titan.training.selection` con otras
paciencias, mejoras mínimas y épocas mínimas, en modo individual y conjunto. El grupo
conjunto es el de la campaña: misma ventana, ámbito, etapa y semilla, con el mismo índice
de caso en la búsqueda o el caso elegido en las finalistas. No entrena ni lee datos.

Un historial termina donde paró su regla original. Si otra regla necesita épocas
posteriores, ese ajuste queda censurado y se cuenta aparte, sin inventar su error.
"""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from mars_titan.training.selection import (
    JOINT_PLATEAU,
    VALIDATION_PLATEAU,
    advance_selection,
    individual_stop,
)

STAGES = ("search", "finalist")


def histories(root):
    """Devuelve los historiales de validación de los ajustes desde cero con su clave de grupo."""
    found, digest = [], hashlib.sha256()
    for path in sorted(Path(root).rglob("run.json")):
        relative = path.relative_to(root).parts
        stage, name = relative[-3], relative[-2]
        if stage not in STAGES:
            continue
        content = path.read_bytes()
        report = json.loads(content)
        if report.get("status") != "completed" or report.get("initial_validation"):
            continue
        digest.update("/".join(relative).encode() + b"\0" + hashlib.sha256(content).digest())
        kind, *rest = name.split("-")
        seed = rest[-1].removeprefix("s")
        case = rest[0] if stage == "search" else "selected"
        epochs = report["epochs"]
        found.append(
            dict(
                key=(relative[0], relative[2], stage, seed, case),
                kind=kind,
                scores=[epoch["validation"]["session_mae"] for epoch in epochs],
                rows=report["samples"]["train"],
                seconds=sum(attempt["seconds"] for attempt in report["attempts"]),
            )
        )
    return found, digest.hexdigest()


def run(scores, options, epochs):
    """Aplica la selección época a época hasta que para el modo o se acaba el historial."""
    state = None
    for epoch, score in enumerate(scores[:epochs], 1):
        state = advance_selection(state, score, epoch, options)
        if state["should_stop"]:
            break
    return state


def best_until(scores, epoch):
    return min(scores[:epoch]) if epoch <= len(scores) else None


def simulate(fits, options, epochs):
    """Resume las épocas recorridas y la pérdida del estado elegido frente a 30 épocas fijas."""
    individual = dict(options, stopping=VALIDATION_PLATEAU)
    joint = dict(options, stopping=JOINT_PLATEAU)
    result = dict(individual=[], joint=[], individual_loss=[], joint_loss=[], censored=0)
    groups = {}
    for fit in fits:
        reference = min(fit["scores"][:epochs])
        state = run(fit["scores"], individual, epochs)
        stop = state["last_epoch"]
        if not state["should_stop"] and stop < epochs:
            result["censored"] += 1
            continue
        result["individual"].append(stop)
        result["individual_loss"].append(state["best_score"] / reference - 1)
        plateau = individual_stop(run(fit["scores"], joint, epochs), epochs)
        groups.setdefault(fit["key"], []).append((fit, plateau))
    for members in groups.values():
        if any(plateau is None for _, plateau in members):
            continue
        common = max(plateau for _, plateau in members)
        for fit, _ in members:
            result["joint"].append(common)
            chosen = best_until(fit["scores"], common)
            if chosen is None:
                result["censored"] += 1
            else:
                result["joint_loss"].append(chosen / min(fit["scores"][:epochs]) - 1)
    return result


def summary(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(
        count=len(values),
        mean=statistics.fmean(values),
        median=statistics.median(values),
        p90=ordered[math.ceil(0.9 * len(ordered)) - 1],
        maximum=ordered[-1],
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, nargs="+", default=[3, 5, 8, 10])
    parser.add_argument("--min-delta", type=float, nargs="+", default=[0.0, 1e-5])
    parser.add_argument("--minimum-epochs", type=int, nargs="+", default=[0, 3, 5, 10])
    args = parser.parse_args(argv)
    fits, digest = histories(args.root)
    rules = []
    for patience in args.patience:
        for delta in args.min_delta:
            for minimum in args.minimum_epochs:
                options = dict(metric="session_mae", patience=patience, min_delta=delta)
                if minimum:
                    options["minimum_epochs"] = minimum
                result = simulate(fits, options, args.epochs)
                rules.append(
                    dict(
                        patience=patience,
                        min_delta=delta,
                        minimum_epochs=minimum,
                        individual_epochs=summary(result["individual"]),
                        joint_epochs=summary(result["joint"]),
                        individual_relative_loss=summary(result["individual_loss"]),
                        joint_relative_loss=summary(result["joint_loss"]),
                        joint_same_selection=sum(v == 0 for v in result["joint_loss"]),
                        individual_same_selection=sum(v == 0 for v in result["individual_loss"]),
                        censored=result["censored"],
                    )
                )
    throughput = [fit["rows"] * len(fit["scores"]) / fit["seconds"] for fit in fits]
    print(
        json.dumps(
            dict(
                schema_version=1,
                kind="early_stopping_history_simulation",
                source=dict(root=args.root.name, runs_sha256=digest),
                fits=len(fits),
                groups=len({fit["key"] for fit in fits}),
                epochs=args.epochs,
                recorded_best_epoch=summary(
                    [fit["scores"].index(min(fit["scores"])) + 1 for fit in fits]
                ),
                recorded_last_epoch=summary([len(fit["scores"]) for fit in fits]),
                fit_seconds=summary([fit["seconds"] for fit in fits]),
                rows_per_second_including_validation=summary(throughput),
                rules=rules,
                final_test_opened=False,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
