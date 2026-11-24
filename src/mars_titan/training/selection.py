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

La métrica de selección es el MAE de sesión del protocolo o, como enmienda que declara la
campaña, el IC de rangos por sesión (`session_rank_ic`). Con errores gaussianos, predecir con
correlación ρ reduce el MAE frente a la predicción cero en un factor √(1 − ρ²) ≈ 1 − ρ²/2: con
ρ = 0,04 son ocho diezmilésimas, así que una mejora mínima de MAE no distingue estados con un
IC muy distinto. Con el IC de rangos la selección minimiza la pérdida (1 − IC) / 2 de la
validación, que está entre 0 y 1.
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
# Un ajuste desde cero puede declarar que su estado inicial se mide en validación pero no se
# emite (#836). Con la mediana inicial en cero ese estado es la predicción cero, y elegirlo
# ocultaría si el modelo ajustado aprende algo. Solo lo admite el ajuste de Titans-MAC.
INITIAL_STATE = "initial_state"
INITIAL_REPORTED = "reported"
# Métricas de selección. El MAE de sesión es la del protocolo. El IC de rangos por sesión es
# la enmienda que puede declarar una campaña.
SESSION_MAE, SESSION_RANK_IC = "session_mae", "session_rank_ic"
METRICS = (SESSION_MAE, SESSION_RANK_IC)
# Definiciones que entran en la identidad de la receta que elige con cada métrica.
MAE_DEFINITION = "SessionErrors.session_mae_on_issued_predictions"
RANK_IC_DEFINITION = (
    "one_minus_mean_session_spearman_of_issued_predictions_and_targets_halved_"
    "over_sessions_with_min_assets_and_two_non_constant_sides_else_one_half"
)
MAX_MIN_ASSETS = 100_000
MAX_SMOOTHING = 127
MAX_LR_REDUCTIONS = 100
MAX_EMIT_AVERAGE = 64
MAX_ADAPTIVE_PINNED = 63
EMIT_BEST = "best"
# Estas son las decisiones posibles tras una validación completa.
CONTINUE, FINISH, AWAIT, DECAY_LR = "continue", "finish", "awaiting_joint_stop", "decay_lr"


def smoothing_window(options):
    """Ventana impar de media móvil centrada de la regla (1 equivale a sin suavizado)."""
    return 1 if options is None else options.get("smoothing", 1)


def lr_decay_config(options):
    """Configuración normalizada de reducción de tasa, o None si está apagada."""
    if options is None or "learning_rate_decay" not in options:
        return None
    raw = options["learning_rate_decay"]
    return dict(factor=float(raw["factor"]), reductions=raw["reductions"])


def emit_average_count(options):
    """Número de mejores épocas que se promedian al emitir (1 equivale a `"best"`)."""
    if options is None or "emit" not in options or options["emit"] == EMIT_BEST:
        return 1
    return options["emit"]["average"]


def has_adaptive_options(options):
    """Si la regla activa suavizado (>1), reducción de tasa o promedio de pesos (>1)."""
    return (
        smoothing_window(options) > 1
        or lr_decay_config(options) is not None
        or emit_average_count(options) > 1
    )


def canonical_selection(options):
    """Copia canónica de la selección sin campos en su valor por defecto.

    Un suavizado igual a 1 o una emisión `"best"` / `{"average": 1}` no alteran el
    comportamiento ni la identidad respecto a omitirlos.
    """
    result = {}
    for key, value in options.items():
        if key == "smoothing":
            if value != 1:
                result[key] = value
        elif key == "emit":
            if value != EMIT_BEST and value != {"average": 1}:
                result[key] = dict(value)
        elif key == "learning_rate_decay":
            result[key] = dict(factor=float(value["factor"]), reductions=value["reductions"])
        else:
            result[key] = value
    return result


def canonical_rule(rule):
    """Copia canónica de una regla completa (con `max_epochs`) omitiendo valores por defecto."""
    result = canonical_selection({key: value for key, value in rule.items() if key != "max_epochs"})
    if "max_epochs" in rule:
        result["max_epochs"] = rule["max_epochs"]
    return result


def _validate_adaptive_options(options):
    if "smoothing" in options:
        window = options["smoothing"]
        if type(window) is not int or not 1 <= window <= MAX_SMOOTHING or window % 2 != 1:
            raise ValueError(f"smoothing debe ser un entero impar entre 1 y {MAX_SMOOTHING}")
    if "learning_rate_decay" in options:
        decay = options["learning_rate_decay"]
        if (
            type(decay) is not dict
            or set(decay) != {"factor", "reductions"}
            or type(decay["factor"]) not in (int, float)
            or not math.isfinite(decay["factor"])
            or not 0.0 < decay["factor"] < 1.0
            or type(decay["reductions"]) is not int
            or not 1 <= decay["reductions"] <= MAX_LR_REDUCTIONS
        ):
            raise ValueError(
                "learning_rate_decay declara factor en (0, 1) y reductions entero entre 1 y 100"
            )
        if options.get("stopping", VALIDATION_PLATEAU) != VALIDATION_PLATEAU:
            raise ValueError("learning_rate_decay solo se admite con la parada validation_plateau")
    if "emit" in options:
        emit = options["emit"]
        valid = emit == EMIT_BEST or (
            type(emit) is dict
            and set(emit) == {"average"}
            and type(emit["average"]) is int
            and 1 <= emit["average"] <= MAX_EMIT_AVERAGE
        )
        if not valid:
            raise ValueError(
                f'emit debe ser "best" o {{"average": k}} con k entero entre 1 y {MAX_EMIT_AVERAGE}'
            )
    window = smoothing_window(options)
    average = emit_average_count(options)
    if options.get("stopping") == COMMON_EPOCH and (window > 1 or average > 1):
        raise ValueError("La época común no se combina con suavizado ni con promedio de pesos")
    if average + (window - 1) // 2 > MAX_ADAPTIVE_PINNED:
        raise ValueError("El suavizado y el promedio de pesos superan el límite de estados fijados")


def validate_selection(
    options, *, epochs=None, modes=STOPPING_MODES, initial_reported=False, adaptive=False
):
    adaptive_keys = ("smoothing", "learning_rate_decay", "emit") if adaptive else ()
    optional = {
        "minimum_epochs",
        "stopping",
        *((INITIAL_STATE,) if initial_reported else ()),
        *adaptive_keys,
    }
    ranked = isinstance(options, dict) and options.get("metric") == SESSION_RANK_IC
    required = {"metric", "patience", "min_delta", *(("min_assets",) if ranked else ())}
    if (
        not isinstance(options, dict)
        or not required <= set(options) <= {*required, *optional}
        or options["metric"] not in METRICS
        or type(options["patience"]) is not int
        or not 1 <= options["patience"] <= 1000
        or type(options["min_delta"]) not in (int, float)
        or not math.isfinite(options["min_delta"])
        or options["min_delta"] < 0
    ):
        raise ValueError(
            "La selección necesita IC de rangos por sesión con sus activos mínimos, paciencia y "
            "mejora mínima válidos"
            if ranked
            else "La selección necesita MAE por sesión, paciencia y mejora mínima válidos"
        )
    if ranked and (
        type(options["min_assets"]) is not int
        or not 3 <= options["min_assets"] <= MAX_MIN_ASSETS
        or options["min_delta"] >= 1
    ):
        raise ValueError(
            "El IC de rangos por sesión declara entre 3 y 100.000 activos mínimos y una mejora "
            "mínima menor que su pérdida máxima"
        )
    if ranked and options.get("stopping") == COMMON_EPOCH:
        raise ValueError(
            "La época común puntúa con su propio IC y conserva el MAE por sesión como métrica"
        )
    if options.get(INITIAL_STATE, INITIAL_REPORTED) != INITIAL_REPORTED:
        raise ValueError("El estado inicial solo se declara como medido y no emitido")
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
    if adaptive:
        _validate_adaptive_options(options)


def metric_definition(options):
    """Definición de la métrica de selección que entra en la identidad de una receta."""
    return RANK_IC_DEFINITION if options["metric"] == SESSION_RANK_IC else MAE_DEFINITION


def selection_score(options, metrics):
    """Puntuación que la selección minimiza con la métrica declarada.

    Con `session_mae` es el MAE por sesión de la validación. Con `session_rank_ic` es la
    pérdida (1 − IC) / 2 del IC de rangos medio de las sesiones con correlación definida, entre
    0 y 1. Una validación sin ninguna sesión definida no ordena los activos y recibe la pérdida
    1/2 de un IC nulo, la misma que una predicción constante.
    """
    if options["metric"] != SESSION_RANK_IC:
        return metrics[SESSION_MAE]
    if SESSION_RANK_IC not in metrics:
        raise ValueError("La validación no se puntuó con el IC de rangos por sesión")
    value = metrics[SESSION_RANK_IC]
    if value is None:
        return 0.5
    if type(value) is not float or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError("El IC de rangos por sesión debe ser un número entre −1 y 1")
    return (1.0 - value) / 2.0


def smoothed_series(values, window):
    """Media móvil centrada de `values` con ventana impar `window`, acotada en los extremos."""
    if type(window) is not int or not 1 <= window <= MAX_SMOOTHING or window % 2 != 1:
        raise ValueError(f"smoothing debe ser un entero impar entre 1 y {MAX_SMOOTHING}")
    if window == 1:
        return [float(value) for value in values]
    radius = (window - 1) // 2
    result = []
    for index in range(len(values)):
        slice_values = values[max(0, index - radius) : index + radius + 1]
        result.append(math.fsum(slice_values) / len(slice_values))
    return result


def _phase_scores(phases, window):
    """Pares (suavizado, crudo) por época en todas las fases de tasa de aprendizaje."""
    scored = {}
    for phase in phases:
        if not phase:
            continue
        smoothed = smoothed_series([raw for _, raw in phase], window)
        for (ep, raw), sm in zip(phase, smoothed, strict=True):
            scored[ep] = (sm, float(raw))
    return scored


def _evaluate_active_phase(cur, *, window, previous, epoch, score, options):
    """Evaluar mejor época y épocas sin mejora de la fase activa con ventana `window`."""
    min_delta = options["min_delta"]
    minimum_epochs = options.get("minimum_epochs", 0)
    anchor_epoch = previous["anchor_best_epoch"]
    anchor_score = previous["anchor_best_score"]
    anchor_raw = previous["anchor_best_raw_score"]

    if window == 1:
        improved = previous["best_score"] is None or score < previous["best_score"] - min_delta
        best_epoch = epoch if improved else previous["best_epoch"]
        best_score = float(score) if improved else previous["best_score"]
        stale = 0 if improved or epoch <= minimum_epochs else previous["stale_epochs"] + 1
        return best_epoch, best_score, best_score, stale, improved, best_score

    radius = (window - 1) // 2
    cur_sm = smoothed_series([raw for _, raw in cur], window)
    if len(cur) <= radius + 1:
        # En las primeras `radius + 1` épocas de la fase todas comparten la ventana completa.
        mean_now = cur_sm[0]
        ref = previous["early_ref_score"]
        if ref is None:
            improved = True
            early_ref = mean_now
            stale = 0
            best_ep, best_raw = min(cur, key=lambda pair: (pair[1], pair[0]))
            return best_ep, mean_now, float(best_raw), stale, improved, early_ref
        improved = mean_now < ref - min_delta
        if improved:
            early_ref = mean_now
            stale = 0
            best_ep, best_raw = min(cur, key=lambda pair: (pair[1], pair[0]))
            return best_ep, mean_now, float(best_raw), stale, True, early_ref
        early_ref = ref
        stale = 0 if epoch <= minimum_epochs else previous["stale_epochs"] + 1
        if anchor_score is not None and not (mean_now < anchor_score - min_delta):
            return anchor_epoch, anchor_score, anchor_raw, stale, False, early_ref
        best_ep, best_raw = min(cur, key=lambda pair: (pair[1], pair[0]))
        return best_ep, mean_now, float(best_raw), stale, False, early_ref

    cur_best_epoch = anchor_epoch
    cur_best_score = anchor_score
    cur_best_raw = anchor_raw
    stale = 0
    step_improved = False
    for (ep, raw), sm in zip(cur, cur_sm, strict=True):
        if ep == 0:
            cur_best_epoch, cur_best_score, cur_best_raw = 0, sm, float(raw)
            continue
        step_improved = cur_best_score is None or sm < cur_best_score - min_delta
        if step_improved:
            cur_best_epoch, cur_best_score, cur_best_raw = ep, sm, float(raw)
            stale = 0
        else:
            stale = 0 if ep <= minimum_epochs else stale + 1
    return (
        cur_best_epoch,
        cur_best_score,
        cur_best_raw,
        stale,
        bool(cur_best_epoch == epoch and step_improved),
        cur_best_score,
    )


def advance_selection(previous, score, epoch, options):
    """Aceptar una mejora estricta y contar solo las épocas ya evaluadas por completo."""
    validate_selection(options, modes=CHRONOLOGICAL_MODES, initial_reported=True, adaptive=True)
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or score < 0
        or type(epoch) is not int
        or not 1 <= epoch <= 1000
    ):
        raise ValueError("El error y la época de selección no son válidos")
    adaptive = has_adaptive_options(options)
    if previous is None:
        previous = dict(
            last_epoch=0, best_epoch=0, best_score=None, stale_epochs=0, should_stop=False
        )
        if adaptive:
            previous.update(
                phases=[[]],
                best_raw_score=None,
                early_ref_score=None,
                anchor_best_epoch=None,
                anchor_best_score=None,
                anchor_best_raw_score=None,
                reductions_done=0,
                pending_decay=False,
            )
    if epoch != previous["last_epoch"] + 1 or previous["should_stop"]:
        raise ValueError("La selección no puede repetir, saltar o continuar épocas ya detenidas")
    if previous.get("pending_decay"):
        raise ValueError("Hay una reducción de tasa pendiente antes de avanzar otra época")

    fixed = options.get("stopping") in (FIXED_BUDGET, JOINT_PLATEAU, COMMON_EPOCH)
    if not adaptive:
        improved = (
            previous["best_score"] is None or score < previous["best_score"] - options["min_delta"]
        )
        stale = (
            0
            if improved or epoch <= options.get("minimum_epochs", 0)
            else previous["stale_epochs"] + 1
        )
        plateau = stale >= options["patience"]
        result = dict(
            last_epoch=epoch,
            best_epoch=epoch if improved else previous["best_epoch"],
            best_score=float(score) if improved else previous["best_score"],
            stale_epochs=stale,
            should_stop=plateau and not fixed,
            last_improved=improved,
        )
        if fixed:
            reached = plateau and options.get("stopping") != COMMON_EPOCH
            earlier = previous.get("plateau_epoch")
            result["plateau_epoch"] = earlier if earlier is not None else epoch if reached else None
        return result

    phases = [list(phase) for phase in previous["phases"]]
    phases[-1].append([epoch, float(score)])
    window = smoothing_window(options)
    best_epoch, best_score, best_raw, stale, improved, early_ref = _evaluate_active_phase(
        phases[-1], window=window, previous=previous, epoch=epoch, score=score, options=options
    )
    plateau = stale >= options["patience"]
    decay = lr_decay_config(options)
    reductions_done = previous["reductions_done"]
    can_decay = decay is not None and reductions_done < decay["reductions"]
    pending_decay = plateau and not fixed and can_decay
    result = dict(
        last_epoch=epoch,
        best_epoch=best_epoch,
        best_score=best_score,
        best_raw_score=best_raw,
        stale_epochs=stale,
        should_stop=plateau and not fixed and not can_decay,
        last_improved=improved,
        phases=phases,
        early_ref_score=early_ref,
        anchor_best_epoch=previous["anchor_best_epoch"],
        anchor_best_score=previous["anchor_best_score"],
        anchor_best_raw_score=previous["anchor_best_raw_score"],
        reductions_done=reductions_done,
        pending_decay=pending_decay,
    )
    if fixed:
        reached = plateau and options.get("stopping") != COMMON_EPOCH
        earlier = previous.get("plateau_epoch")
        result["plateau_epoch"] = earlier if earlier is not None else epoch if reached else None
    return result


def apply_lr_decay(selection, options):
    """Registrar una reducción de tasa, abrir una fase nueva y reiniciar la paciencia."""
    validate_selection(options, modes=CHRONOLOGICAL_MODES, initial_reported=True, adaptive=True)
    decay = lr_decay_config(options)
    if (
        decay is None
        or not isinstance(selection, dict)
        or not selection.get("pending_decay")
        or selection.get("reductions_done", 0) >= decay["reductions"]
    ):
        raise ValueError("No hay una reducción de tasa de aprendizaje pendiente y admitida")
    phases = [list(phase) for phase in selection["phases"]]
    phases.append([])
    return dict(
        selection,
        phases=phases,
        stale_epochs=0,
        should_stop=False,
        last_improved=False,
        early_ref_score=selection["best_score"],
        anchor_best_epoch=selection["best_epoch"],
        anchor_best_score=selection["best_score"],
        anchor_best_raw_score=selection["best_raw_score"],
        reductions_done=selection["reductions_done"] + 1,
        pending_decay=False,
    )


def initial_selection(score, options):
    """Registrar el padre antes del ajuste sin consumir una época ni paciencia.

    Con `initial_state: reported` su puntuación queda en la historia del ajuste, pero no
    compite: la primera época ajustada lo sustituye siempre como mejor estado.
    """
    state = advance_selection(None, score, 1, options)
    state.update(last_epoch=0, best_epoch=0)
    reported = options.get(INITIAL_STATE) == INITIAL_REPORTED
    if reported:
        state["best_score"] = None
    if has_adaptive_options(options):
        state["phases"] = [[]] if reported else [[[0, float(score)]]]
        state["best_raw_score"] = None if reported else float(score)
        state["early_ref_score"] = None if reported else float(score)
        state["pending_decay"] = False
        state["should_stop"] = False
        state["stale_epochs"] = 0
    return state


def selected_emit_epochs(selection, options):
    """Épocas elegidas para emitir: `[best_epoch]` o las `k` mejores según la métrica."""
    best = selection["best_epoch"]
    count = emit_average_count(options)
    if count == 1 or "phases" not in selection:
        return [best]
    scored = _phase_scores(selection["phases"], smoothing_window(options))
    if not scored:
        return [best]
    ranked = sorted(scored, key=lambda ep: (scored[ep][0], scored[ep][1], ep))
    if best in scored:
        ordered = [best, *[ep for ep in ranked if ep != best]]
    else:
        ordered = ranked
    return ordered[: min(count, len(ordered))]


def retained_epochs(selection, options):
    """Épocas cuyos puntos de control deben seguir fijados para suavizado y promedio."""
    if "phases" not in selection:
        return [selection["best_epoch"]] if selection["best_score"] is not None else []
    phases = selection["phases"]
    window = smoothing_window(options)
    radius = (window - 1) // 2
    count = emit_average_count(options)
    scored = _phase_scores(phases, window)
    finalized = []
    for phase_index, phase in enumerate(phases):
        limit = len(phase) if phase_index < len(phases) - 1 else max(0, len(phase) - radius)
        for ep, _ in phase[:limit]:
            finalized.append(ep)
    keep = set(sorted(finalized, key=lambda ep: (scored[ep][0], scored[ep][1], ep))[:count])
    active = phases[-1] if phases else []
    for ep, _ in active[max(0, len(active) - radius) :]:
        keep.add(ep)
    if selection["best_score"] is not None:
        keep.add(selection["best_epoch"])
    return sorted(keep)


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
    """Decidir tras una validación completa: seguir, terminar, reducir tasa o esperar.

    `selection` es None antes de la primera validación de un ajuste sin estado inicial. Con
    `joint_plateau` y sin `joint_epoch`, un ajuste espera en su primera meseta o, si no la
    alcanza, al agotar el máximo. Con `joint_epoch` recorre exactamente hasta esa época, que
    no puede ser anterior a su propia parada ni a la última época evaluada, ni superar el
    máximo. Con `common_epoch`, `joint_epoch` es la época común del grupo y elige uno de los
    estados ya evaluados. Con `learning_rate_decay`, al agotarse la paciencia antes del tope de
    épocas y con reducciones pendientes devuelve `DECAY_LR`.
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
    if selection.get("pending_decay"):
        return DECAY_LR
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
    selection = canonical_selection(
        {key: value for key, value in rule.items() if key != "max_epochs"}
    )
    if recipe.epochs == rule["max_epochs"] and canonical_selection(recipe.selection) == selection:
        return recipe
    return replace(recipe, epochs=rule["max_epochs"], selection=selection)


def campaign_rule(
    protocol_rule, override, *, modes=STOPPING_MODES, initial_reported=False, adaptive=False
):
    """Devuelve la regla de un ajuste, que es la del protocolo o la parada temprana de la campaña.

    La campaña puede cambiar el modo, la paciencia, la mejora mínima, las épocas mínima y
    máxima y, en los ajustes que lo admiten, el suavizado, la reducción de tasa y la emisión
    promediada. La métrica solo cambia en un sentido: una campaña puede declarar el IC de rangos
    por sesión en lugar del MAE del protocolo. Los valores por defecto de las opciones nuevas se
    omiten de la forma canónica para conservar la identidad exacta del comportamiento base.
    """
    if override is None:
        return canonical_rule(protocol_rule)
    if (
        not isinstance(override, dict)
        or override.get("metric") not in (protocol_rule["metric"], SESSION_RANK_IC)
        or (
            protocol_rule["metric"] == SESSION_RANK_IC
            and override.get("min_assets") != protocol_rule.get("min_assets")
        )
        or "max_epochs" not in override
        or type(override["max_epochs"]) is not int
        or not 1 <= override["max_epochs"] <= 1000
    ):
        raise ValueError(
            "La parada de la campaña conserva la métrica, o declara el IC de rangos, y su máximo"
        )
    validate_selection(
        {key: value for key, value in override.items() if key != "max_epochs"},
        epochs=override["max_epochs"],
        modes=modes,
        initial_reported=initial_reported,
        adaptive=adaptive,
    )
    return canonical_rule(override)


def _metric_from_score(metric, score):
    if score is None:
        return None
    return 1.0 - 2.0 * score if metric == SESSION_RANK_IC else score


def build_stopping_report(
    history,
    selection,
    options,
    *,
    epochs,
    stopped_early,
    emitted_epochs=None,
    lr_decays=None,
    learning_rates=None,
    best_epoch=None,
):
    """Construir el informe por ajuste con curva completa de entrenamiento/validación y diagnóstico.

    Registra por época las métricas de entrenamiento y de validación (cruda y suavizada), la
    tasa activa, las reducciones de tasa, los pesos emitidos y un diagnóstico explícito:
    si paró por meseta, cuántas épocas transcurrieron después del máximo y la diferencia entre
    la métrica de entrenamiento y la de validación en la mejor época.
    """
    metric = options["metric"]
    higher_is_better = metric == SESSION_RANK_IC
    window = smoothing_window(options)
    stopping_mode = options.get("stopping", VALIDATION_PLATEAU)
    best = selection["best_epoch"] if best_epoch is None else best_epoch
    decays = [] if lr_decays is None else [dict(item) for item in lr_decays]
    lrs = {} if learning_rates is None else dict(learning_rates)
    emitted = (
        selected_emit_epochs(selection, options) if emitted_epochs is None else list(emitted_epochs)
    )
    emitted_set = set(emitted)

    if "phases" in selection:
        phases = selection["phases"]
    else:
        reported = options.get(INITIAL_STATE) == INITIAL_REPORTED
        phases = [
            [
                [entry["epoch"], selection_score(options, entry["validation"])]
                for entry in history
                if not (reported and entry["epoch"] == 0)
            ]
        ]
    scored = _phase_scores(phases, window)
    epoch_phase = {}
    for phase_index, phase in enumerate(phases):
        for ep, _ in phase:
            epoch_phase[ep] = phase_index

    curve = []
    for entry in history:
        ep = entry["epoch"]
        train = entry.get("train") or {}
        validation = entry.get("validation") or {}
        val_score = selection_score(options, validation)
        sm_score = scored[ep][0] if ep in scored else None
        sm_metric = _metric_from_score(metric, sm_score)
        train_metric = train.get(metric) if entry.get("train") is not None else None
        val_metric = validation.get(metric)
        if train_metric is not None and val_metric is not None:
            gap = (
                float(train_metric - val_metric)
                if higher_is_better
                else float(val_metric - train_metric)
            )
        else:
            gap = None
        curve.append(
            dict(
                epoch=ep,
                global_step=entry.get("global_step"),
                phase=epoch_phase.get(ep),
                learning_rate=lrs.get(ep),
                train_metric=train_metric,
                train_mae=train.get(SESSION_MAE) if entry.get("train") is not None else None,
                train_mse=train.get("session_mse") if entry.get("train") is not None else None,
                train_loss=train.get("mean_loss") if entry.get("train") is not None else None,
                validation_metric=val_metric,
                validation_mae=validation.get(SESSION_MAE),
                validation_mse=validation.get("session_mse"),
                validation_score=float(val_score),
                validation_smoothed_score=sm_score,
                validation_smoothed_metric=sm_metric,
                train_validation_gap=gap,
                selected=ep == best,
                in_emitted_average=ep in emitted_set,
            )
        )

    final_epoch = history[-1]["epoch"] if history else 0
    best_row = next((row for row in curve if row["epoch"] == best), None)
    best_train = None if best_row is None else best_row["train_metric"]
    best_val = None if best_row is None else best_row["validation_metric"]
    best_val_sm = None if best_row is None else best_row["validation_smoothed_metric"]
    best_diff = None if best_train is None or best_val is None else float(best_train - best_val)
    best_gap = None if best_row is None else best_row["train_validation_gap"]
    best_train_mae = None if best_row is None else best_row["train_mae"]
    best_val_mae = None if best_row is None else best_row["validation_mae"]
    best_mae_diff = (
        None
        if best_train_mae is None or best_val_mae is None
        else float(best_train_mae - best_val_mae)
    )
    stopped_by_plateau = bool(selection.get("should_stop"))
    if stopped_by_plateau:
        stop_reason = VALIDATION_PLATEAU
    elif stopping_mode in (COMMON_EPOCH, JOINT_PLATEAU):
        stop_reason = stopping_mode
    else:
        stop_reason = "max_epochs"

    emit_count = emit_average_count(options)
    return dict(
        schema_version=1,
        kind="fit_stopping_report_v1",
        metric=metric,
        higher_is_better=higher_is_better,
        stopping_mode=stopping_mode,
        max_epochs=epochs,
        patience=options["patience"],
        min_delta=options["min_delta"],
        smoothing=window,
        learning_rate_decay=lr_decay_config(options),
        emit=EMIT_BEST if emit_count == 1 else dict(average=emit_count),
        curve=curve,
        learning_rate_reductions=decays,
        emitted=dict(
            mode="average" if len(emitted) > 1 else EMIT_BEST,
            epochs=emitted,
            best_epoch=best,
            memories_and_online_state_epoch=best,
        ),
        diagnosis=dict(
            stopped_by_plateau=stopped_by_plateau,
            stopped_early=bool(stopped_early),
            stop_reason=stop_reason,
            reached_max_epochs=final_epoch >= epochs,
            final_epoch=final_epoch,
            best_epoch=best,
            epochs_after_best=final_epoch - best,
            learning_rate_reductions_used=len(decays),
            best_epoch_train_metric=best_train,
            best_epoch_validation_metric=best_val,
            best_epoch_validation_smoothed_metric=best_val_sm,
            best_epoch_train_validation_difference=best_diff,
            best_epoch_generalization_gap=best_gap,
            best_epoch_train_mae=best_train_mae,
            best_epoch_validation_mae=best_val_mae,
            best_epoch_mae_difference=best_mae_diff,
        ),
    )


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
