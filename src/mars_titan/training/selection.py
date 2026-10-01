"""Selección de épocas por MAE de sesión, sin consultar la reserva final."""

import math


def validate_selection(options, *, epochs=None):
    if (
        not isinstance(options, dict)
        or not {"metric", "patience", "min_delta"}
        <= set(options)
        <= {"metric", "patience", "min_delta", "minimum_epochs"}
        or options["metric"] != "session_mae"
        or type(options["patience"]) is not int
        or not 1 <= options["patience"] <= 1000
        or type(options["min_delta"]) not in (int, float)
        or not math.isfinite(options["min_delta"])
        or options["min_delta"] < 0
    ):
        raise ValueError("La selección necesita MAE por sesión, paciencia y mejora mínima válidos")
    if "minimum_epochs" in options and (
        type(options["minimum_epochs"]) is not int
        or not 0 <= options["minimum_epochs"] < 1000
        or (epochs is not None and options["minimum_epochs"] >= epochs)
    ):
        raise ValueError(
            "El mínimo de épocas debe dejar evaluaciones posteriores en el presupuesto"
        )


def advance_selection(previous, score, epoch, options):
    """Aceptar una mejora estricta y contar solo las épocas ya evaluadas por completo."""
    validate_selection(options)
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or score < 0
        or type(epoch) is not int
        or not 1 <= epoch <= 1000
    ):
        raise ValueError("El error y la época de selección no son válidos")
    if previous is None:
        previous = dict(
            last_epoch=0, best_epoch=0, best_score=None, stale_epochs=0, should_stop=False
        )
    if epoch != previous["last_epoch"] + 1 or previous["should_stop"]:
        raise ValueError("La selección no puede repetir, saltar o continuar épocas ya detenidas")
    improved = (
        previous["best_score"] is None or score < previous["best_score"] - options["min_delta"]
    )
    stale = (
        0 if improved or epoch <= options.get("minimum_epochs", 0) else previous["stale_epochs"] + 1
    )
    return dict(
        last_epoch=epoch,
        best_epoch=epoch if improved else previous["best_epoch"],
        best_score=float(score) if improved else previous["best_score"],
        stale_epochs=stale,
        should_stop=stale >= options["patience"],
        last_improved=improved,
    )


def initial_selection(score, options):
    """Registrar el padre antes del ajuste sin consumir una época ni paciencia."""
    state = advance_selection(None, score, 1, options)
    state.update(last_epoch=0, best_epoch=0)
    return state
