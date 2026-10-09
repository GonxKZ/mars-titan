"""Ablación de modalidades en inferencia para la comparación walk-forward.

Es un análisis secundario y descriptivo, declarado el 9 de octubre de 2026 antes de
cualquier resultado. No sirve para elegir modelos ni cambia la conclusión principal.
Mide cuánto depende cada brazo de noticias y fundamentales en las mismas filas: el estado
elegido de cada brazo, semilla y ventana vuelve a predecir la evaluación con la modalidad
leída como una ausencia real (`data.modality_ablation`), sin reentrenar ni recalibrar.

La métrica es la diferencia del MAE residual por sesión entre la predicción enmascarada y
la original, emparejada fila a fila. Solo cuentan las filas con alguna de las modalidades
ablacionadas presente, que son las únicas cuya entrada cambia. Las demás se cuentan y, en
un modelo sin memoria, deben conservar su predicción. La cobertura y la anchura de los
intervalos usan el calibrador común ya ajustado con las predicciones originales.

No mide un efecto causal económico de la modalidad ni su utilidad. Un modelo puede
depender de una modalidad, es decir, cambiar su predicción sin ella, y no ganar nada con
ella. Tampoco equivale a reentrenar sin la modalidad.
"""

import dataclasses
from datetime import date

import numpy as np

from mars_titan.data.input_policy import MODALITIES
from mars_titan.data.modality_ablation import REASON
from mars_titan.data.modality_ablation import VARIANTS as MASKED
from mars_titan.evaluation import modality_strata
from mars_titan.evaluation.forecast_panel import SessionSeries
from mars_titan.evaluation.forecast_scores import SessionScores, score_sessions
from mars_titan.evaluation.paired_comparisons import compare_series, delta
from mars_titan.models.quantile_head import QUANTILE_HEAD

STATUS = "secondary_descriptive"
USE = "description_only_not_for_model_selection_or_primary_conclusion"
VARIANTS = {name: list(modalities) for name, modalities in MASKED.items()}
MASKING = "mask_contract_v1_presence_false_missing_fill_null_availability"
STATE = "selected_state_of_each_arm_seed_and_window_without_refitting"
MEMORY = "warmup_and_measured_partition_read_the_same_masked_inputs"
ROWS = "evaluation_rows_with_any_masked_modality_present"
METRIC = "session_mae_masked_minus_original"
MULTIPLICITY = "bonferroni_over_variants_and_views_with_max_t_over_arms"
DOES_NOT_MEASURE = [
    "economic_causal_effect_of_the_modality",
    "usefulness_of_the_modality_for_the_forecast",
    "performance_of_a_model_trained_without_the_modality",
]
FIELDS = {
    "status",
    "declared_at",
    "use",
    "variants",
    "masking",
    "missing_reason",
    "state",
    "memory",
    "partition",
    "rows",
    "metric",
    "recalibrate",
    "min_rows",
    "min_sessions",
    "multiplicity",
    "does_not_measure",
}
SOURCES_KIND = "walk_forward_ablation_sources"
NO_ROWS = "Ninguna fila evaluada de la celda tiene las modalidades ablacionadas"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(section):
    """Validar la declaración previa de la ablación sin abrir ningún dato."""
    _require(
        isinstance(section, dict) and set(section) == FIELDS,
        "La declaración de la ablación no tiene exactamente sus campos",
    )
    declared = section["declared_at"]
    try:
        valid_date = isinstance(declared, str) and date.fromisoformat(declared).isoformat()
    except ValueError:
        valid_date = False
    _require(
        section["status"] == STATUS
        and section["use"] == USE
        and valid_date == declared
        and section["multiplicity"] == MULTIPLICITY
        and section["does_not_measure"] == DOES_NOT_MEASURE,
        "La ablación debe declararse como análisis secundario y descriptivo con sus límites",
    )
    _require(
        section["variants"] == VARIANTS
        and section["masking"] == MASKING
        and section["missing_reason"] == REASON
        and section["state"] == STATE
        and section["memory"] == MEMORY,
        "Las variantes enmascaran noticias, fundamentales o ambas como una ausencia real",
    )
    _require(
        section["partition"] == "evaluation"
        and section["rows"] == ROWS
        and section["metric"] == METRIC,
        "La métrica es la diferencia del MAE por sesión en las filas con la modalidad",
    )
    _require(section["recalibrate"] is False, "La ablación no puede volver a calibrar")
    for name in ("min_rows", "min_sessions"):
        value = section[name]
        _require(
            type(value) is int and 1 <= value <= modality_strata.MAX_THRESHOLD,
            f"El umbral {name} debe ser un entero positivo fijado de antemano",
        )
    return section


def affected_rows(bits, variant):
    """Filas con alguna modalidad de la variante presente, las únicas cuya entrada cambia."""
    columns = [MODALITIES.index(name) for name in VARIANTS[variant]]
    return bits[:, columns].any(axis=1)


def score_pair(original, masked, rows, *, rank_ic_min_assets, calibrated=(None, None)):
    """Puntuar la predicción original y la enmascarada en las filas afectadas.

    Los dos paneles tienen las mismas filas, objetivos y orden canónico. `calibrated` son
    los cuantiles de cada uno corregidos por el calibrador común de la ventana, que no se
    vuelve a ajustar. Se cuentan además las filas sin la modalidad cuya predicción cambia,
    que en un modelo sin memoria deben ser cero.
    """
    _require(
        original.cohort_sha256 == masked.cohort_sha256 and rows.shape == (original.rows,),
        "La predicción enmascarada no evalúa las mismas filas que la original",
    )
    unaffected = ~rows
    entry = dict(
        rows=int(rows.sum()),
        unaffected_rows=int(unaffected.sum()),
        unaffected_changed=int(
            np.count_nonzero(original.prediction[unaffected] != masked.prediction[unaffected])
        ),
        original=None,
        masked=None,
        calibrated_original=None,
        calibrated_masked=None,
    )
    if not rows.any():
        return entry
    for name, panel, quantiles in zip(
        ("original", "masked"), (original, masked), calibrated, strict=True
    ):
        part = modality_strata._subset(panel, rows)
        entry[name] = score_sessions(part, rank_ic_min_assets=rank_ic_min_assets)
        if quantiles is not None:
            adjusted = np.ascontiguousarray(quantiles[rows])
            adjusted.setflags(write=False)
            entry[f"calibrated_{name}"] = score_sessions(
                dataclasses.replace(part, quantiles=adjusted),
                rank_ic_min_assets=rank_ic_min_assets,
            )
    return entry


def _cell(scores, thresholds, market=None):
    cell = modality_strata.cell(scores, thresholds, market)
    if cell["rows"] == 0:
        cell["reason"] = NO_ROWS
    return cell


def _joined(parts, name, markets, names):
    present = [part[name] for part in parts if part[name] is not None]
    return modality_strata.subset_views(
        SessionScores.concatenate(present) if present else None, markets, names
    )


def _summary(views, quantile, cell, weighting):
    """MAE original, enmascarado y su diferencia, e intervalos calibrados de una celda."""
    result = dict(
        original_session_mae=None,
        masked_session_mae=None,
        difference=None,
        reason=cell["reason"],
        calibrated_intervals=None,
        calibrated_reason=None,
    )
    if not cell["estimable"]:
        result["calibrated_reason"] = cell["reason"]
        return result
    original, masked = (
        views[name].summary(market_weighting=weighting)["point"]["mae"]
        for name in ("original", "masked")
    )
    result.update(original_session_mae=original, masked_session_mae=masked)
    result["difference"] = masked - original
    if not quantile:
        result["calibrated_reason"] = "El brazo no emite cuantiles"
    elif views["calibrated_original"] is None:
        result["calibrated_reason"] = "Alguna ventana no tiene calibrador"
    else:
        result["calibrated_intervals"] = {
            name: [
                {key: row[key] for key in ("nominal", "coverage", "coverage_gap", "width")}
                for row in views[f"calibrated_{name}"].summary(market_weighting=weighting)[
                    "quantiles"
                ]["intervals"]
            ]
            for name in ("original", "masked")
        }
    return result


def report(config, per_window, markets, names, *, sources_sha256):
    """Sección `modality_ablation` del informe con la declaración, recuentos y contrastes.

    `per_window` asigna a cada ventana y (brazo, semilla) las puntuaciones de cada variante
    de `score_pair`. Una celda por debajo de los umbrales aparece con su motivo y sin
    métricas. Los contrastes de cada variante y vista forman una familia sobre los brazos
    con máximo estudentizado, y la confianza corrige por Bonferroni el número de variantes
    por vistas.
    """
    declared = config["modality_ablation"]
    comparison = config["comparison"]
    weighting = config["metrics"]["market_weighting"]
    thresholds = {key: declared[key] for key in ("min_rows", "min_sessions")}
    cells = len(VARIANTS) * len(names)
    confidence = modality_strata.adjusted_confidence(comparison["confidence"], cells)
    windows = list(per_window)
    keys = list(per_window[windows[0]])
    variants = {}
    for variant, modalities in VARIANTS.items():
        by_key = {}
        for key in keys:
            parts = [per_window[window][key][variant] for window in windows]
            views = {
                name: _joined(parts, name, markets, names)
                for name in ("original", "masked", "calibrated_original", "calibrated_masked")
            }
            # Sin calibrador en alguna ventana, la celda tampoco tiene cobertura calibrada.
            if any(part["rows"] and part["calibrated_original"] is None for part in parts):
                views["calibrated_original"] = views["calibrated_masked"] = dict.fromkeys(names)
            by_key[key] = views, parts
        first_views, first_parts = by_key[keys[0]]
        reference = first_views["original"]
        whole = {view: _cell(reference[view], thresholds) for view in names}
        population = dict(
            overall=whole,
            windows={
                window: dict(
                    rows=part["rows"],
                    unaffected_rows=part["unaffected_rows"],
                    markets={
                        market: _cell(part["original"], thresholds, market) for market in markets
                    },
                )
                for window, part in zip(windows, first_parts, strict=True)
            },
        )
        arms, series = {}, {}
        for (arm, seed), (views, parts) in by_key.items():
            quantile = config["arms"][arm]["output"] == QUANTILE_HEAD
            arms.setdefault(arm, {})[str(seed)] = dict(
                overall={
                    view: _summary(
                        {name: value[view] for name, value in views.items()},
                        quantile,
                        whole[view],
                        weighting,
                    )
                    for view in names
                },
                unaffected_changed={
                    window: part["unaffected_changed"]
                    for window, part in zip(windows, parts, strict=True)
                },
            )
            series.setdefault(arm, []).append(views)
        contrasts = {}
        for view in names:
            if not whole[view]["estimable"]:
                contrasts[view] = dict(reason=whole[view]["reason"])
                continue
            pairs = {}
            for arm, items in series.items():
                for name in ("original", "masked"):
                    pairs[f"{arm}/{name}"] = SessionSeries.average(
                        [item[name][view].series("mae") for item in items]
                    )
            contrasts[view] = compare_series(
                pairs,
                {arm: delta(f"{arm}/original", f"{arm}/masked") for arm in series},
                block_length=comparison["block_length"],
                replicates=comparison["replicates"],
                seed=comparison["seed"],
                confidence=confidence,
                market_weighting=weighting,
                sensitivity_block_lengths=tuple(comparison["sensitivity_block_lengths"]),
            )
        variants[variant] = dict(
            modalities=modalities, population=population, arms=arms, contrasts=contrasts
        )
    return dict(
        declaration=declared,
        status="computed",
        reason=None,
        sources_sha256=sources_sha256,
        recalibrated=False,
        multiplicity=dict(
            method=MULTIPLICITY,
            cells=cells,
            family_confidence=comparison["confidence"],
            confidence=confidence,
        ),
        variants=variants,
    )


def pending(config):
    """Sección declarada cuya etapa todavía no ha publicado predicciones enmascaradas."""
    return dict(
        declaration=config["modality_ablation"],
        status="not_computed",
        reason="Faltan las predicciones enmascaradas de la etapa de ablación",
    )
