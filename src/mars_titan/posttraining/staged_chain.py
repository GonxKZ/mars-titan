"""Contrato de la cadena del walk-forward por etapas que publica la etapa de adaptadores.

En la ventana k ≥ 1 de un ámbito, para cada brazo base y semilla, el predictor de la
cadena es el candidato con menor puntuación de validación de esa ventana entre el padre
congelado (el estado elegido de la campaña base en k-1), cada caso de adaptadores de la
matriz y la continuación completa del padre con las filas nuevas. Solo sustituye al padre
congelado con una mejora estricta y los demás empates se resuelven por el identificador
del trabajo (`chain_validation_score_v1`). En la ventana 0 el predictor es el estado
elegido de la base.

La puntuación es el MAE medio por sesión de la mediana sobre todas las filas de `val_k`, la
misma definición que elige el estado de la campaña base. Se recalcula con
`validation_score` sobre las predicciones de validación guardadas de cada candidato, que
salen del mismo recorrido, así que dos candidatos con las mismas predicciones empatan.

Los identificadores, rutas, regla y lectura reproducen `training.campaign_chain` del plan
de la campaña y se sustituirán por ese módulo cuando esté en `develop`.
"""

import math
from pathlib import Path

import numpy as np
import pyarrow as pa

from mars_titan.evaluation.session_metrics import SessionErrors

CHAIN_SUFFIX = "__chain"
FROZEN_SUFFIX = "__frozen_parent"
RULE = "chain_validation_score_v1"
SELECTION = "selection.json"
SELECTION_KIND = "campaign_chain_selection"
CANDIDATES = ("frozen_parent", "adapter", "continuation")
SELECTED = ("base", *CANDIDATES)
_SELECTION = {
    "kind",
    "schema_version",
    "campaign_sha256",
    "stage_sha256",
    "scope",
    "window",
    "base_arm",
    "seed",
    "rule",
    "parent_window",
    "parent",
    "candidates",
    "selected",
    "state",
    "fit_rows",
    "markets",
    "labels_used_until",
    "confirmed_at_utc",
}
# Filas por actualización del acumulador por sesión.
_BATCH = 4096


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def chain_arm(base_arm):
    """Brazo publicado del predictor de la cadena de un brazo base."""
    return f"{base_arm}{CHAIN_SUFFIX}"


def frozen_arm(base_arm):
    """Brazo de las predicciones del padre congelado de un brazo base."""
    return f"{base_arm}{FROZEN_SUFFIX}"


def chain_job_id(scope, window, base_arm, seed):
    """Trabajo de selección de la cadena de un ámbito, ventana, brazo base y semilla."""
    return f"{scope}/{window}/{chain_arm(base_arm)}/select-s{seed}"


def chain_folder(root, scope, window, base_arm, seed):
    """Carpeta de los recibos de la cadena, bajo la salida de la etapa de posentrenamiento."""
    return Path(root) / "windows" / scope / window / chain_arm(base_arm) / f"seed-{seed}"


def scope_windows(campaign, scope):
    """Ventanas de un ámbito en orden temporal, con sus tramos."""
    windows = campaign["comparison_config"]["resolved_scopes"][scope]["windows"]
    ordered = sorted(windows.items(), key=lambda item: item[1]["evaluation"][0])
    _require(
        [name for name, _ in ordered] == list(windows),
        f"Las ventanas de {scope} no están en orden temporal",
    )
    return ordered


def parent_jobs(base_jobs, scope, window, arm, seed):
    """Trabajos base que eligen el estado de un brazo y semilla en una ventana.

    Un traslado o un finalista de esa semilla, o las búsquedas de la semilla de búsqueda.
    """
    prefix = f"{scope}/{window}/{arm}/"
    found = {job["id"]: job for job in base_jobs if job["id"].startswith(prefix)}
    for stage in ("carry", "finalist"):
        if f"{prefix}{stage}-s{seed}" in found:
            return [f"{prefix}{stage}-s{seed}"]
    searches = sorted(
        key for key, job in found.items() if job["stage"] == "search" and job["seed"] == seed
    )
    _require(searches, f"La campaña base no elige el estado de {prefix} con la semilla {seed}")
    return searches


def validation_score(table):
    """MAE medio por sesión de la predicción (la mediana) sobre todas las filas.

    Las filas se recorren en orden de mercado, activo e instante, así que el valor solo
    depende de las predicciones y no del orden en que se escribieron.
    """
    market = table["market"].to_numpy(zero_copy_only=False).astype(str)
    asset = table["asset_id"].to_numpy(zero_copy_only=False).astype(str)
    moment = table["prediction_at"].cast(pa.int64()).to_numpy()
    error = table["prediction"].to_numpy().astype(np.float64) - table["target"].to_numpy()
    order = np.lexsort((moment, asset, market))
    errors = SessionErrors()
    for start in range(0, len(order), _BATCH):
        rows = order[start : start + _BATCH]
        errors.update(market[rows], moment[rows], error[rows])
    score = errors.summary()["session_mae"]
    _require(
        isinstance(score, float) and math.isfinite(score) and score >= 0,
        "La validación no tiene un MAE por sesión finito",
    )
    return score


def choose(candidates):
    """Candidato elegido por `chain_validation_score_v1`.

    Gana el menor `score` de validación. El padre congelado solo se sustituye con una
    mejora estricta y los empates entre los demás se resuelven por identificador del trabajo.
    """
    _require(
        all(
            candidate["kind"] in CANDIDATES
            and isinstance(candidate["score"], (int, float))
            and math.isfinite(candidate["score"])
            for candidate in candidates
        )
        and len({candidate["job"] for candidate in candidates}) == len(candidates)
        and [c["kind"] for c in candidates].count("frozen_parent") == 1,
        "La cadena compara un padre congelado y candidatos distintos con score finito",
    )
    frozen = next(c for c in candidates if c["kind"] == "frozen_parent")
    others = sorted(
        (c for c in candidates if c["kind"] != "frozen_parent"),
        key=lambda c: (c["score"], c["job"]),
    )
    return others[0] if others and others[0]["score"] < frozen["score"] else frozen


def read_selection(root, scope, window, base_arm, seed):
    """Selección confirmada de la cadena con sus recibos por mercado, o None si no existe.

    Exige la regla, la coherencia entre ventana, padre, candidatos y filas nuevas, y que cada
    recibo de mercado conserve su huella, el contrato #390, el trabajo elegido y la última
    etiqueta usada.
    """
    from mars_titan.data.cohort_files import read_manifest
    from mars_titan.data.storage import sha256
    from mars_titan.environments.walk_forward_receipt import read_window_receipt

    folder = chain_folder(root, scope, window, base_arm, seed)
    path = folder / SELECTION
    if not path.is_file():
        return None
    document, digest = read_manifest(path, 4 * 1024**2)
    label = f"{scope}/{window}/{chain_arm(base_arm)}/seed-{seed}"
    _require(
        isinstance(document, dict)
        and set(document) == _SELECTION
        and document["kind"] == SELECTION_KIND
        and document["schema_version"] == 1
        and document["rule"] == RULE
        and (document["scope"], document["window"]) == (scope, window)
        and (document["base_arm"], document["seed"]) == (base_arm, seed),
        f"La selección de {label} no cumple su contrato",
    )
    selected = document["selected"]
    if document["parent_window"] is None:
        _require(
            selected["kind"] == "base"
            and document["parent"] is None
            and document["candidates"] == []
            and document["fit_rows"] is None,
            f"La primera ventana de {label} elige el estado de la base sin padre",
        )
    else:
        chosen = choose(document["candidates"])
        _require(
            selected["kind"] == chosen["kind"]
            and selected["job"] == chosen["job"]
            and selected["receipt_sha256"] == chosen["receipt_sha256"]
            and isinstance(document["parent"], dict)
            and (document["fit_rows"] is None) == (selected["kind"] == "frozen_parent"),
            f"La selección de {label} no sigue la regla {RULE}",
        )
    markets = document["markets"]
    _require(isinstance(markets, dict) and markets, f"La selección de {label} no tiene mercados")
    receipts = {}
    for market, expected in sorted(markets.items()):
        receipt_path = folder / f"{market}.json"
        _require(
            receipt_path.is_file() and sha256(receipt_path) == expected,
            f"El recibo {market} de {label} no corresponde a su selección",
        )
        receipt = read_window_receipt(read_manifest(receipt_path, 1024**2)[0])
        _require(
            receipt.market == market
            and receipt.parent == (selected["job"], selected["receipt_sha256"])
            and receipt.labels_used_until == document["labels_used_until"],
            f"El recibo {market} de {label} no es el del predictor elegido",
        )
        receipts[market] = receipt
    return dict(document, sha256=digest, receipts=receipts)
