"""Selección de épocas por MAE de sesión, sin consultar la reserva final.

Hay cuatro modos de parada. `fixed_budget` recorre todas las épocas y conserva el mejor
estado. `validation_plateau` para cada ajuste en su primera meseta. `joint_plateau` es la
parada conjunta de un grupo de controles emparejados: cada ajuste se detiene en su primera
meseta en un estado recuperable (`AWAIT`) y continúa después hasta la época común del grupo,
de modo que todos aplican el mismo número de actualizaciones y de validaciones.

`common_epoch` es la regla de la enmienda de la criba (#746). Cada ajuste recorre todo el
presupuesto, conserva el estado de cada época evaluada y espera (`AWAIT`) a que el grupo de
semillas fije la época común con `common_epoch`. Las semillas aplican así las mismas
actualizaciones y emiten la misma época. El MAE de sesión sigue eligiendo el mejor estado del
propio ajuste, que solo se informa como sensibilidad.
"""

import math
from dataclasses import replace

import numpy as np

VALIDATION_PLATEAU = "validation_plateau"
# Recorre todas las épocas declaradas y conserva el mejor estado sin cortar el presupuesto.
FIXED_BUDGET = "fixed_budget"
# Este modo detiene a la vez un grupo emparejado. La época común la fija la campaña.
JOINT_PLATEAU = "joint_plateau"
STOPPING_MODES = (VALIDATION_PLATEAU, FIXED_BUDGET, JOINT_PLATEAU)
# Este modo recorre el presupuesto y emite la época que elige el grupo de semillas. Solo lo
# admiten los entrenadores que lo implementan: Titans-MAC y el lector de MARS-TITAN.
COMMON_EPOCH = "common_epoch"
CHRONOLOGICAL_MODES = (*STOPPING_MODES, COMMON_EPOCH)
# Estas son las decisiones posibles tras una validación completa.
CONTINUE, FINISH, AWAIT = "continue", "finish", "awaiting_joint_stop"


def validate_selection(options, *, epochs=None, modes=STOPPING_MODES):
    if (
        not isinstance(options, dict)
        or not {"metric", "patience", "min_delta"}
        <= set(options)
        <= {"metric", "patience", "min_delta", "minimum_epochs", "stopping"}
        or options["metric"] != "session_mae"
        or type(options["patience"]) is not int
        or not 1 <= options["patience"] <= 1000
        or type(options["min_delta"]) not in (int, float)
        or not math.isfinite(options["min_delta"])
        or options["min_delta"] < 0
    ):
        raise ValueError("La selección necesita MAE por sesión, paciencia y mejora mínima válidos")
    if "stopping" in options and options["stopping"] not in modes:
        raise ValueError(
            "La parada debe ser meseta de validación, presupuesto fijo o meseta conjunta. "
            "La época común solo la admiten Titans-MAC y el lector de MARS-TITAN"
        )
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
    validate_selection(options, modes=CHRONOLOGICAL_MODES)
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
    plateau = stale >= options["patience"]
    # La meseta conjunta tampoco corta sola: la época común la decide el grupo.
    fixed = options.get("stopping") in (FIXED_BUDGET, JOINT_PLATEAU, COMMON_EPOCH)
    result = dict(
        last_epoch=epoch,
        best_epoch=epoch if improved else previous["best_epoch"],
        best_score=float(score) if improved else previous["best_score"],
        stale_epochs=stale,
        should_stop=plateau and not fixed,
        last_improved=improved,
    )
    if fixed:
        # Registrar dónde habría parado la meseta, sin cambiar el número de actualizaciones.
        # La época común no tiene meseta: su ajuste siempre recorre todo el presupuesto.
        reached = plateau and options.get("stopping") != COMMON_EPOCH
        earlier = previous.get("plateau_epoch")
        result["plateau_epoch"] = earlier if earlier is not None else epoch if reached else None
    return result


def initial_selection(score, options):
    """Registrar el padre antes del ajuste sin consumir una época ni paciencia."""
    state = advance_selection(None, score, 1, options)
    state.update(last_epoch=0, best_epoch=0)
    return state


def individual_stop(selection, epochs):
    """Devuelve la primera meseta de un ajuste conjunto, o el máximo de épocas si lo agotó.

    Devuelve None mientras el ajuste no haya alcanzado ninguna de las dos.
    """
    if selection is None:
        return None
    if selection.get("plateau_epoch") is not None:
        return selection["plateau_epoch"]
    return epochs if selection["last_epoch"] >= epochs else None


def epoch_decision(selection, options, epochs, joint_epoch=None):
    """Decidir tras una validación completa: seguir, terminar o esperar la época conjunta.

    `selection` es None antes de la primera validación de un ajuste sin estado inicial. Con
    `joint_plateau` y sin `joint_epoch`, un ajuste espera en su primera meseta o, si no la
    alcanza, al agotar el máximo. Con `joint_epoch` recorre exactamente hasta esa época, que
    no puede ser anterior a su propia parada ni a la última época evaluada, ni superar el
    máximo. Con `common_epoch`, `joint_epoch` es la época común del grupo y elige uno de los
    estados ya evaluados. Los demás modos no la admiten.
    """
    last = 0 if selection is None else selection["last_epoch"]
    stopping = None if options is None else options.get("stopping")
    if stopping == COMMON_EPOCH:
        return _common_decision(last, epochs, joint_epoch)
    joint = stopping == JOINT_PLATEAU
    if joint_epoch is not None:
        stop = individual_stop(selection, epochs)
        if (
            not joint
            or type(joint_epoch) is not int
            or stop is None
            or not max(last, stop) <= joint_epoch <= epochs
        ):
            raise ValueError(
                "La época conjunta solo existe en la meseta conjunta y debe estar entre la "
                "parada del ajuste y el máximo de épocas"
            )
        return FINISH if last >= joint_epoch else CONTINUE
    if joint:
        # Sin época común, la meseta siempre espera al grupo, también al agotar el máximo.
        return AWAIT if individual_stop(selection, epochs) is not None else CONTINUE
    if last >= epochs:
        return FINISH
    if selection is None:
        return CONTINUE
    return FINISH if selection["should_stop"] else CONTINUE


def _common_decision(last, epochs, common):
    """Seguir hasta agotar el presupuesto, esperar al grupo y terminar con la época común.

    La época común elige uno de los estados ya evaluados, así que solo se acepta cuando el
    ajuste ha validado todas sus épocas.
    """
    if common is None:
        return AWAIT if last >= epochs else CONTINUE
    if type(common) is not int or not 1 <= common <= epochs or last < epochs:
        raise ValueError(
            "La época común llega después de evaluar todo el presupuesto y está entre 1 y "
            "el máximo de épocas"
        )
    return FINISH


def bind_joint_epoch(report, joint_epoch, options, epochs):
    """Fijar en el informe la época conjunta recibida. Una reanudación no puede cambiarla.

    La época se valida con la selección confirmada del informe antes de fijarla, así que
    una época no válida no llega a escribirse.
    """
    recorded = report.get("joint_stop_epoch")
    if recorded is not None and joint_epoch != recorded:
        raise ValueError("La ejecución ya recibió otra época conjunta")
    if joint_epoch is not None:
        epoch_decision(report.get("selection"), options, epochs, joint_epoch)
        report["joint_stop_epoch"] = joint_epoch


def awaiting(selection, epochs):
    """Devuelve los campos del informe de un ajuste detenido en su meseta a la espera del grupo."""
    return dict(
        status=AWAIT,
        individual_stop_epoch=individual_stop(selection, epochs),
        plateau_epoch=selection["plateau_epoch"],
        best_epoch=selection["best_epoch"],
        best_score=selection["best_score"],
        last_epoch=selection["last_epoch"],
    )


def with_rule(recipe, rule):
    """Devuelve la receta con la regla de parada de la campaña, que solo cambia épocas y selección.

    La receta declara la regla del protocolo. Una campaña con parada temprana declarada
    sustituye solo esos dos campos, así que la identidad de la receta cambia con el modo.
    """
    selection = {key: value for key, value in rule.items() if key != "max_epochs"}
    if recipe.epochs == rule["max_epochs"] and recipe.selection == selection:
        return recipe
    return replace(recipe, epochs=rule["max_epochs"], selection=selection)


def campaign_rule(protocol_rule, override, *, modes=STOPPING_MODES):
    """Devuelve la regla de un ajuste, que es la del protocolo o la parada temprana de la campaña.

    La campaña puede cambiar el modo, la paciencia, la mejora mínima y las épocas mínima y
    máxima, nunca la métrica. Sin `override` se devuelve la del protocolo. `modes` son los
    modos que admite el entrenador que la aplica.
    """
    if override is None:
        return dict(protocol_rule)
    if (
        not isinstance(override, dict)
        or override.get("metric") != protocol_rule["metric"]
        or "max_epochs" not in override
        or type(override["max_epochs"]) is not int
        or not 1 <= override["max_epochs"] <= 1000
    ):
        raise ValueError("La parada de la campaña conserva la métrica y declara su máximo")
    validate_selection(
        {key: value for key, value in override.items() if key != "max_epochs"},
        epochs=override["max_epochs"],
        modes=modes,
    )
    return dict(override)


def seed_epoch_scores(series):
    """IC medio del grupo en cada época, con las sesiones que definen todas las semillas.

    `series[s][e - 1]` es el par (valores, definidas) del IC por sesión de la semilla s en la
    época e. En cada época se promedian las semillas sesión a sesión y la media recorre las
    sesiones que todas definen, igual que la serie de un modelo en el análisis de la criba.
    Una época sin ninguna sesión común definida no tiene puntuación y detiene la selección.
    """
    if not series or len({len(epochs) for epochs in series}) != 1 or not series[0]:
        raise ValueError("Todas las semillas del grupo deben evaluar las mismas épocas")
    scores = []
    for epoch in range(len(series[0])):
        values = [np.asarray(item[epoch][0], dtype=np.float64) for item in series]
        defined = [np.asarray(item[epoch][1], dtype=bool) for item in series]
        if len({value.shape for value in values + defined}) != 1 or values[0].ndim != 1:
            raise ValueError("Las semillas del grupo no comparten las sesiones de validación")
        common = np.logical_and.reduce(defined)
        if not common.any():
            raise ValueError(f"La época {epoch + 1} no define el IC en ninguna sesión común")
        mean = np.mean(values, axis=0)[common]
        if not np.isfinite(mean).all():
            raise ValueError(f"La época {epoch + 1} tiene un IC no finito")
        scores.append(float(mean.mean()))
    return scores


def common_epoch(scores, *, radius=1):
    """Época común de un grupo: la de mayor puntuación suavizada con sus vecinas.

    `scores[e - 1]` es la puntuación del grupo en la época e, mayor es mejor. Cada época se
    puntúa con la media de las épocas a distancia `radius` o menos dentro del presupuesto. Un
    empate se resuelve con la época más temprana, la que aplica menos actualizaciones.
    """
    if (
        not scores
        or any(type(value) not in (int, float) or not math.isfinite(value) for value in scores)
        or type(radius) is not int
        or radius < 0
    ):
        raise ValueError("La época común necesita puntuaciones finitas y un radio no negativo")
    smoothed = []
    for index in range(len(scores)):
        window = scores[max(0, index - radius) : index + radius + 1]
        smoothed.append(math.fsum(window) / len(window))
    best = max(range(len(scores)), key=lambda index: (smoothed[index], -index))
    return dict(epoch=best + 1, scores=list(scores), smoothed=smoothed, radius=radius)
