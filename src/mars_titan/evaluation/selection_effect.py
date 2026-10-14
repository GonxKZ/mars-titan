"""Efecto de elegir entre los casos de búsqueda con la validación de cada ventana.

La campaña ajusta cada caso de búsqueda con la semilla de búsqueda y elige el de menor MAE
por sesión de validación. Todos los casos confirman además sus predicciones de evaluación,
así que el efecto de esa elección se mide sin ejecutar nada: se puntúan todos los casos en la
evaluación de su ventana con la métrica de la comparación y se sitúa el elegido entre ellos.

Por ventana y modelo, con n casos y el MAE de evaluación de cada uno:

- rango del elegido de 1 (peor MAE) a n (mejor), con rango medio en los empates;
- rango relativo ω = rango / (n + 1) y su logit λ = ln(ω / (1 − ω)), como en Bailey,
  Borwein, López de Prado y Zhu (2017). λ ≤ 0 indica que el elegido no supera la mediana;
- ganancia sobre la mediana de los casos y distancia al mejor caso de la evaluación, que es
  una cota con información futura y no una alternativa disponible al elegir.

Por modelo se cuenta en cuántas ventanas λ ≤ 0. Es la versión por ventanas de la
probabilidad de sobreajuste de la selección: cada ventana separa la validación de la
evaluación en orden temporal, en lugar de las combinaciones de bloques de CSCV. No hay
p-valor: las ventanas comparten activos y régimen y no son ensayos independientes.

Solo se miden los modelos que puntúa la comparación del ámbito, porque la métrica y las filas
son las suyas. Un modelo con búsqueda que la comparación no declara, como las variantes de
CM-v1 en A v2, queda fuera con su motivo en cada ventana y en el informe.
"""

import math
from pathlib import Path

import numpy as np

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import walk_forward_comparison as comparison
from mars_titan.evaluation.forecast_scores import score_sessions
from mars_titan.models.quantile_head import QUANTILE_COLUMNS

KIND = "search_selection_effect"
REPORT_KIND = "search_selection_effect_report"
WINDOW_KIND = "search_selection_effect_window"
# Código que determina el efecto guardado de una ventana.
CODE = (
    "evaluation/selection_effect.py",
    "evaluation/walk_forward_comparison.py",
    "evaluation/forecast_panel.py",
    "evaluation/forecast_scores.py",
)
# Lo que mide este módulo. La declaración debe coincidir para que el informe la cite.
DECLARED = dict(
    schema_version=1,
    kind=KIND,
    selection="min_validation_session_mae_then_job_id",
    partition="evaluation",
    metric="primary_metric_of_the_comparison_with_its_market_weighting",
    minimum_cases=2,
    statistics=["relative_rank", "logit", "gain_over_median", "gap_to_best_in_hindsight"],
    aggregate="windows_with_logit_at_or_below_zero",
)
NOT_DECLARED = "not_declared_in_the_comparison"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_declaration(path):
    """Leer la declaración previa y exigir que describa lo que mide este módulo."""
    declaration, digest = read_manifest(Path(path), 1024**2)
    _require(isinstance(declaration, dict), "La declaración debe ser un objeto JSON")
    _require(
        {key: declaration.get(key) for key in DECLARED} == DECLARED,
        "La declaración no coincide con el efecto de la selección que se mide",
    )
    _require(declaration.get("final_test_opened") is False, "La declaración abre la reserva")
    reasons = declaration.get("not_applied")
    _require(
        isinstance(reasons, dict) and all(isinstance(v, str) and v for v in reasons.values()),
        "La declaración debe explicar los análisis que no aplica",
    )
    return declaration, digest


def window_effect(scores, selected):
    """Posición del caso elegido entre los MAE de evaluación de todos los casos.

    `scores` asigna a cada trabajo su MAE de evaluación. Con menos de dos casos no hay
    elección que medir y el resultado lo dice.
    """
    _require(selected in scores, "El caso elegido no está entre los casos puntuados")
    values = np.array([scores[key] for key in sorted(scores)], dtype=np.float64)
    _require(bool(np.all(np.isfinite(values))), "Hay un MAE de evaluación no finito")
    cases = len(values)
    if cases < DECLARED["minimum_cases"]:
        return dict(cases=cases, applicable=False, reason="Un solo caso: no hay elección")
    chosen = scores[selected]
    # Rango 1 para el peor MAE y n para el mejor. Los empates comparten su rango medio.
    worse, ties = int(np.sum(values > chosen)), int(np.sum(values == chosen))
    rank = 1 + worse + (ties - 1) / 2
    omega = rank / (cases + 1)
    median = float(np.median(values))
    return dict(
        cases=cases,
        applicable=True,
        selected=selected,
        selected_mae=chosen,
        median_mae=median,
        best_mae=float(values.min()),
        rank=rank,
        relative_rank=omega,
        logit=math.log(omega / (1 - omega)),
        gain_over_median=median - chosen,
        gap_to_best_in_hindsight=chosen - float(values.min()),
    )


def arm_summary(windows):
    """Ventanas con elección, cuántas dejan al elegido en la mediana o por debajo y medias."""
    measured = [item for item in windows.values() if item["applicable"]]
    if not measured:
        return dict(windows=0, at_or_below_median=None, share_at_or_below_median=None)
    below = sum(item["logit"] <= 0 for item in measured)
    return dict(
        windows=len(measured),
        at_or_below_median=below,
        share_at_or_below_median=below / len(measured),
        median_logit=float(np.median([item["logit"] for item in measured])),
        mean_gain_over_median=float(np.mean([item["gain_over_median"] for item in measured])),
        mean_gap_to_best_in_hindsight=float(
            np.mean([item["gap_to_best_in_hindsight"] for item in measured])
        ),
    )


def case_scores(config, scope, window_id, arm, cases):
    """MAE de evaluación de cada caso, puntuado como en la comparación y sobre las mismas filas."""
    resolved = config["resolved_scopes"][scope]
    declared = resolved["arms"][arm]
    quantile = declared["output"] == comparison.QUANTILE_HEAD
    columns = comparison.COLUMNS + (QUANTILE_COLUMNS if quantile else ())
    window = resolved["windows"][window_id]
    weighting = config["metrics"]["market_weighting"]
    minimum = config["metrics"]["rank_ic_min_assets"]
    result, reference = {}, None
    for key in sorted(cases):
        label = f"{key} en la evaluación"
        table = comparison._read_predictions(cases[key]["evaluation"], columns)
        table, _ = comparison._restricted(table, resolved, window_id, arm)
        panel = comparison._panel(
            table, window, "evaluation", resolved, label, output=declared["output"]
        )
        reference = reference or (label, panel)
        comparison._same_rows(reference, panel, label)
        summary = score_sessions(panel, rank_ic_min_assets=minimum).summary(
            market_weighting=weighting
        )
        result[key] = summary["point"][config["metrics"]["primary"]]
    return result


def window_effects(config, scope, window_id, models):
    """Efecto de la elección de cada modelo con búsqueda en una ventana.

    `models` asigna a cada modelo el trabajo elegido y sus casos, como los devuelve
    `masked_campaign.search_records`. Junto a la evaluación queda la validación que decidió.
    Devuelve los efectos de los modelos que puntúa la comparación (`arms`) y, aparte, los
    modelos con búsqueda que no declara, con su motivo (`excluded`).
    """
    _require(config["metrics"]["primary"] == "mae", "La comparación no usa el MAE por sesión")
    declared = config["resolved_scopes"][scope]["arms"]
    result = {}
    for arm, record in sorted(models.items()):
        if arm not in declared:
            continue
        scores = case_scores(config, scope, window_id, arm, record["cases"])
        effect = window_effect(scores, record["selected"])
        effect["validation"] = {key: case["score"] for key, case in record["cases"].items()}
        effect["evaluation"] = scores
        effect["predictions"] = {
            key: case["evaluation"]["sha256"] for key, case in record["cases"].items()
        }
        result[arm] = effect
    excluded = {arm: NOT_DECLARED for arm in sorted(models) if arm not in declared}
    return dict(arms=result, excluded=excluded)


def _identity(config, scope, window_id):
    root = Path(comparison.__file__).parents[1]
    return dict(
        comparison_sha256=config["sha256"],
        scope=scope,
        window=window_id,
        code={name: sha256(root / name) for name in CODE},
    )


def path_for(folder, scope, window_id):
    return Path(folder) / scope / f"{window_id}.selection.json"


def write_window(folder, config, scope, window_id, models):
    """Guardar el efecto de la elección de una ventana antes de liberar sus predicciones."""
    path = path_for(folder, scope, window_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = dict(
        schema_version=1,
        kind=WINDOW_KIND,
        identity=_identity(config, scope, window_id),
        **window_effects(config, scope, window_id, models),
        final_test_opened=False,
    )
    atomic_json(path, document)
    return dict(path=str(path), sha256=sha256(path), bytes=path.stat().st_size)


def read_window(folder, config, scope, window_id):
    """Efecto guardado de una ventana, si procede de esta comparación y de este código."""
    path = path_for(folder, scope, window_id)
    _require(path.is_file(), f"Falta el efecto de la selección de {scope} {window_id}")
    document, _ = read_manifest(path, 16 * 1024**2)
    _require(
        isinstance(document, dict)
        and document.get("kind") == WINDOW_KIND
        and document.get("identity") == _identity(config, scope, window_id)
        and document.get("final_test_opened") is False,
        f"El efecto de la selección de {scope} {window_id} no procede de esta comparación",
    )
    return dict(arms=document["arms"], excluded=document["excluded"])


def selection_report(scope, windows, declaration, declaration_sha256):
    """Informe por modelo a partir del efecto de cada ventana, sin leer predicciones."""
    arms, excluded = {}, {}
    for window_id, effects in sorted(windows.items()):
        for arm, effect in effects["arms"].items():
            arms.setdefault(arm, {})[window_id] = effect
        for arm, reason in effects["excluded"].items():
            entry = excluded.setdefault(arm, dict(reason=reason, windows=[]))
            entry["windows"].append(window_id)
    return dict(
        schema_version=1,
        kind=REPORT_KIND,
        scope=scope,
        declaration_sha256=declaration_sha256,
        not_applied=declaration["not_applied"],
        excluded=excluded,
        final_test_opened=False,
        arms={
            arm: dict(windows=effects, summary=arm_summary(effects))
            for arm, effects in sorted(arms.items())
        },
    )
