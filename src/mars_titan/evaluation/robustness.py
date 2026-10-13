"""Robustez ante perturbaciones de las entradas en la comparación walk-forward (#37).

Es un análisis secundario y descriptivo, como la ablación de modalidades. La declaración
vive fuera de la configuración de la comparación
(`configs/evaluation/historical-masked-2000-robustness.json`) para no cambiar su huella. La
etapa de robustez (`training.modality_ablation_stage` con su propio tipo) vuelve a predecir
la evaluación de cada ventana con el estado elegido de cada modelo, semilla y ventana y la
entrada perturbada en la lectura (`data.input_perturbations`), sin reentrenar ni recalibrar.
El calentamiento y el tramo medido leen la misma entrada perturbada.

Todo se empareja fila a fila con la predicción original del mismo estado:
- degradación: diferencia del MAE residual por sesión en las filas cuya entrada cambia,
  con los mismos contrastes que la ablación (`modality_ablation.variant_report`);
- deriva: en las filas cuya entrada no cambia, cuántas predicciones cambian y cuánto. Un
  modelo sin estado ni contexto debe conservarlas. Uno con memoria en línea o con historia
  de varias sesiones arrastra la perturbación;
- recuperación tras una ráfaga: esa deriva y la diferencia de error por días desde el final
  de la última ráfaga del activo, en los tramos declarados.

No mide la robustez ante datos reales distintos de estas perturbaciones, ni la utilidad de
una modalidad, ni el comportamiento de un modelo reentrenado con entradas perturbadas.
"""

import numpy as np

from mars_titan.data.input_perturbations import (
    BURST,
    PERTURBATIONS,
    burst_rows,
    days_since_burst,
)
from mars_titan.data.input_policy import MODALITIES

from . import modality_ablation, robustness_protocol

DECLARATION = robustness_protocol.ROOT / "configs/evaluation/historical-masked-2000-robustness.json"
SOURCES_KIND = "walk_forward_robustness_sources"
VARIANTS = {name: list(spec["modalities"]) for name, spec in PERTURBATIONS.items()}
BURSTS = tuple(name for name, spec in PERTURBATIONS.items() if spec["kind"] == BURST)
STATUS = modality_ablation.STATUS
USE = modality_ablation.USE
METRIC = "session_mae_perturbed_minus_original"
ROWS = "evaluation_rows_whose_input_changes"
DRIFT = "unchanged_rows_by_days_since_the_last_burst_of_the_asset"
MULTIPLICITY = modality_ablation.MULTIPLICITY
ANALYSIS_FIELDS = {
    "declared_at",
    "status",
    "use",
    "metric",
    "rows",
    "drift",
    "recovery_bins_days",
    "min_rows",
    "min_sessions",
    "multiplicity",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def declaration(path=DECLARATION, root=robustness_protocol.ROOT):
    """Declaración de robustez con su análisis, comprobada contra el repositorio."""
    document, digest = robustness_protocol.load_declaration(path, root)
    section = document.get("analysis")
    _require(
        isinstance(section, dict) and set(section) == ANALYSIS_FIELDS,
        "La declaración de robustez no tiene exactamente los campos de su análisis",
    )
    _require(
        section["status"] == STATUS
        and section["use"] == USE
        and section["metric"] == METRIC
        and section["rows"] == ROWS
        and section["drift"] == DRIFT
        and section["multiplicity"] == MULTIPLICITY,
        "La robustez es un análisis secundario y descriptivo con su métrica y sus filas",
    )
    bins = section["recovery_bins_days"]
    _require(
        isinstance(bins, list)
        and bins
        and all(type(value) is int and value >= 1 for value in bins)
        and bins == sorted(set(bins)),
        "Los tramos de recuperación deben ser días enteros crecientes",
    )
    for name in ("min_rows", "min_sessions"):
        value = section[name]
        _require(
            type(value) is int and value >= 1,
            f"El umbral {name} debe ser un entero positivo fijado de antemano",
        )
    # La ruta no entra: la identidad de la etapa no puede depender de dónde está el clon.
    return dict(section, sha256=digest)


def burst_context(assets, moments, horizon_days):
    """Ráfagas y días desde la última de cada fila, por variante de ráfaga.

    `assets` son identificadores `mercado/símbolo`, como la clave de la lectura, y
    `moments` los instantes de predicción en microsegundos UTC, en el orden del panel.
    """
    assets = np.asarray(assets)
    moments = np.asarray(moments, dtype=np.int64)
    _require(assets.shape == moments.shape, "Activos e instantes no tienen las mismas filas")
    names, inverse = np.unique(assets, return_inverse=True)
    _require(all("/" in str(name) for name in names), "Los activos deben ser mercado/símbolo")
    order = np.argsort(inverse, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(inverse[order])) + 1, len(order)]
    context = {}
    for variant in BURSTS:
        spec = PERTURBATIONS[variant]
        inside, since = np.zeros(len(moments), dtype=bool), np.full(len(moments), np.nan)
        for index, name in enumerate(names):
            rows = order[starts[index] : starts[index + 1]]
            arguments = (str(name), moments[rows], spec["length_days"], spec["share"])
            inside[rows] = burst_rows(*arguments)
            since[rows] = days_since_burst(*arguments, horizon_days)
        context[variant] = inside, since
    return context


def affected_rows(bits, variant, bursts):
    """Filas cuya entrada cambia: alguna modalidad de la variante presente y, en una
    ráfaga, además dentro de ella."""
    present = bits[:, [MODALITIES.index(name) for name in VARIANTS[variant]]].any(axis=1)
    if variant in BURSTS:
        return present & bursts[variant][0]
    return present


def drift(original, perturbed, rows, since, bins):
    """Deriva en las filas cuya entrada no cambia, por tramos de días desde la última ráfaga.

    Guarda sumas, no medias, para agregar ventanas sumando. El último tramo, `far`, reúne las
    filas sin ráfaga anterior dentro del horizonte o fuera de ellas por no tener la modalidad.
    """
    unchanged = ~rows
    difference = np.abs(perturbed.prediction - original.prediction)
    error = np.abs(original.target - perturbed.prediction) - np.abs(
        original.target - original.prediction
    )
    edges = [0, *bins]
    selections = [
        (f"{low}-{high}", unchanged & (since >= low) & (since < high))
        for low, high in zip(edges[:-1], edges[1:], strict=True)
    ]
    selections.append(("far", unchanged & np.isnan(since)))
    return {
        name: dict(
            rows=int(selected.sum()),
            changed=int(np.count_nonzero(difference[selected])),
            sum_abs_difference=float(difference[selected].sum()),
            sum_error_difference=float(error[selected].sum()),
        )
        for name, selected in selections
    }


def score(original, perturbed, variant, bits, bursts, bins, *, rank_ic_min_assets, calibrated):
    """Puntuaciones de una variante: las de la ablación y, en una ráfaga, su deriva."""
    rows = affected_rows(bits, variant, bursts)
    entry = modality_ablation.score_pair(
        original, perturbed, rows, rank_ic_min_assets=rank_ic_min_assets, calibrated=calibrated
    )
    if variant in BURSTS:
        entry["drift"] = drift(original, perturbed, rows, bursts[variant][1], bins)
    return entry


def _drift_summary(parts):
    totals = {}
    for part in parts:
        for name, values in part.items():
            total = totals.setdefault(name, dict.fromkeys(values, 0))
            for key, value in values.items():
                total[key] += value
    return {
        name: dict(
            rows=total["rows"],
            changed=total["changed"],
            mean_abs_difference=total["sum_abs_difference"] / total["rows"]
            if total["rows"]
            else None,
            mean_error_difference=total["sum_error_difference"] / total["rows"]
            if total["rows"]
            else None,
        )
        for name, total in totals.items()
    }


def report(declared, config, per_window, markets, names, *, sources_sha256):
    """Sección `robustness` del informe: degradación, contrastes y deriva por tramo."""
    section = modality_ablation.variant_report(
        declared, VARIANTS, config, per_window, markets, names, sources_sha256=sources_sha256
    )
    windows = list(per_window)
    for variant in BURSTS:
        arms = section["variants"][variant]["arms"]
        for arm, seed in per_window[windows[0]]:
            parts = [per_window[window][arm, seed][variant]["drift"] for window in windows]
            arms[arm][str(seed)]["drift"] = _drift_summary(parts)
    return section
