"""Walk-forward por etapas de la campaña A: roles de cada ventana y contrato de la cadena.

Decisión del autor del 9 de octubre de 2026 («Opción 1 + control»): ninguna etapa ajusta
pesos con filas cuyos objetivos ya usó la etapa anterior para ajustar los suyos. Para la
ventana k de un ámbito, con evaluación en el año Y:

1. `base`: la campaña base ajusta con `train_k` = [2000, abr Y-1), elige con `val_k`
   (abr a sep Y-1), calibra con `cal_k` (oct a dic Y-1) y predice `eval_k` (año Y).
2. `posttraining` (k ≥ 1): parte del estado elegido de la base en la ventana k-1 para el
   mismo brazo y semilla, que ajustó hasta marzo de Y-2 y usó abr a dic de Y-2 para elegir
   y calibrar. Solo ajusta con las filas de `train_k` cuya decisión cae entre el final de
   `cal_{k-1}` y el final de `train_k` (ene a mar de Y-1), elige con `val_k` y calibra con
   `cal_k`. Candidatos: el padre congelado, los casos de adaptadores y la continuación
   completa del padre con esas filas. La ventana 0 no tiene padre.
3. `chain`: el predictor de la cadena es el candidato con menor `score` de validación de
   la ventana k. Solo sustituye al padre congelado con una mejora estricta. En la ventana
   0 es el estado elegido de la base. El reentreno completo (la base k) no es candidato:
   la cadena mide la adaptación con datos que el padre no vio, y la comparación base k
   frente a cadena k se informa aparte, con las mismas filas de test.
4. `rl`: la política anclada en k ajusta con las cintas de evaluación de todas las
   ventanas anteriores a su validación (mínimo 3), valida con la de k-1 y evalúa con la de
   k. Cada cinta lleva las predicciones del predictor de la cadena de su ventana.
5. `test`: `eval_k`, igual para todas las familias.

Este módulo declara esos roles, los identificadores de los trabajos de la cadena, las rutas
de sus recibos y las dependencias que el calendario exige a las etapas. No lee vistas ni
ajusta nada. `chain_disjunction` comprueba las filas sobre las vistas y los recibos.
"""

import hashlib
import math
from pathlib import Path

import numpy as np

CHAIN_SUFFIX = "__chain"
RULE = "chain_validation_score_v1"
SELECTION = "selection.json"
SELECTION_KIND = "campaign_chain_selection"
CANDIDATES = ("frozen_parent", "adapter", "continuation")
SELECTED = ("base", *CANDIDATES)
MIN_RL_TRAIN_WINDOWS = 3
# Declaración única admitida en `walk_forward_stages` de la campaña.
DESIGN = dict(
    design="staged_chain_v1",
    posttraining=dict(
        parent="base_selected_previous_window",
        fit_rows="after_parent_calibration_until_train_end",
        selection="validation",
        calibration="calibration",
        candidates=list(CANDIDATES),
        reported_apart=["base_retrain"],
    ),
    chain=dict(rule=RULE, min_improvement=0.0, first_window="base"),
    rl=dict(
        train="expanding_previous_evaluations",
        min_train_windows=MIN_RL_TRAIN_WINDOWS,
        validation="previous_evaluation",
        evaluation="own_evaluation",
        predictor="chain",
    ),
    test="evaluation",
)
FINGERPRINT = b"mars-titan-chain-rows-v1"
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


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _same(value, expected):
    """Igualdad que distingue tipos: 0 no es 0.0 ni False es 0."""
    if isinstance(expected, dict):
        return (
            isinstance(value, dict)
            and set(value) == set(expected)
            and all(_same(value[k], v) for k, v in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(value, list)
            and len(value) == len(expected)
            and all(_same(a, b) for a, b in zip(value, expected, strict=True))
        )
    return type(value) is type(expected) and value == expected


def declared(value):
    """Validar la declaración de la campaña. Solo se admite el diseño por etapas fijado."""
    _require(
        _same(value, DESIGN),
        "La campaña declara walk_forward_stages con el diseño staged_chain_v1 completo",
    )
    return value


def chain_arm(base_arm):
    """Brazo publicado del predictor de la cadena de un brazo base."""
    return f"{base_arm}{CHAIN_SUFFIX}"


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


def parent_window(campaign, scope, window):
    """Ventana del padre del posentrenamiento, o None en la primera ventana del ámbito."""
    names = [name for name, _ in scope_windows(campaign, scope)]
    _require(window in names, f"{window} no es una ventana de {scope}")
    index = names.index(window)
    return names[index - 1] if index else None


def posttraining_rows(parent_fold, fold):
    """Intervalo [inicio, fin) de decisiones de las filas nuevas del posentrenamiento.

    Empieza al final de la calibración del padre, después de todo lo que el padre usó para
    ajustar, elegir o calibrar, y termina con el tramo de ajuste de la ventana. Las filas
    son las del tramo `train` de la vista de la ventana, cuya etiqueta ya madura antes de
    ese final, así que la purga queda en los dos extremos.
    """
    # La calibración es el último tramo que el padre usa: ajuste y validación acaban antes.
    start, end = parent_fold["calibration"][1], fold["train"][1]
    _require(
        fold["train"][0] <= start < end,
        f"{fold['id']} no tiene filas nuevas después del padre {parent_fold['id']}",
    )
    return start, end


def window_roles(campaign, scope, window):
    """Roles declarados de una ventana de un ámbito, con sus intervalos de decisiones."""
    folds = dict(scope_windows(campaign, scope))
    fold = folds[window]
    parent = parent_window(campaign, scope, window)
    base = {name: list(fold[name]) for name in ("train", "validation", "calibration")}
    roles = dict(window=window, scope=scope, base=dict(base, evaluation=list(fold["evaluation"])))
    if parent is None:
        roles["posttraining"] = None
        roles["chain"] = dict(rule=RULE, candidates=["base"])
    else:
        roles["posttraining"] = dict(
            parent_window=parent,
            parent="base_selected_state",
            fit=list(posttraining_rows(folds[parent], fold)),
            validation=list(fold["validation"]),
            calibration=list(fold["calibration"]),
            evaluation=list(fold["evaluation"]),
        )
        roles["chain"] = dict(rule=RULE, candidates=list(CANDIDATES))
    roles["rl"] = rl_windows(list(folds), window)
    roles["test"] = list(fold["evaluation"])
    return roles


def rl_windows(windows, window, minimum=MIN_RL_TRAIN_WINDOWS):
    """Ventanas de la política anclada en `window`, o None si no hay bastantes.

    `windows` son las ventanas en orden en las que el mercado tiene evaluación. La política
    ajusta con todas las anteriores a su validación, valida con la anterior y evalúa la suya.
    """
    _require(window in windows, f"{window} no tiene evaluación en ese mercado")
    index = windows.index(window)
    if index - 1 < minimum:
        return None
    return dict(train=windows[: index - 1], validation=windows[index - 1], evaluation=window)


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


def chain_jobs(campaign, base_jobs, adapter_jobs):
    """Selecciones de la cadena de cada brazo base y semilla de la etapa de posentrenamiento.

    En la ventana 0 dependen de los trabajos base que eligen su estado. En las demás, de
    todos los trabajos de posentrenamiento de su ámbito, ventana, brazo base y semilla.
    """
    keys = sorted({(job["scope"], job["base_arm"], job["seed"]) for job in adapter_jobs})
    by_key = {}
    for job in adapter_jobs:
        by_key.setdefault((job["scope"], job["window"], job["base_arm"], job["seed"]), []).append(
            job["id"]
        )
    jobs = []
    for scope, base_arm, seed in keys:
        for index, (window, _) in enumerate(scope_windows(campaign, scope)):
            if index == 0:
                depends = parent_jobs(base_jobs, scope, window, base_arm, seed)
            else:
                depends = by_key.get((scope, window, base_arm, seed), [])
                _require(depends, f"{scope}/{window}/{base_arm} no tiene posentrenamiento")
            jobs.append(
                dict(
                    id=chain_job_id(scope, window, base_arm, seed),
                    scope=scope,
                    window=window,
                    anchor=window,
                    arm=chain_arm(base_arm),
                    base_arm=base_arm,
                    seed=seed,
                    kind="chain",
                    stage="chain",
                    depends=depends,
                )
            )
    return jobs


def check_staged(campaign, base_jobs, stages):
    """Exigir a las etapas posteriores las dependencias del diseño por etapas.

    Un trabajo de posentrenamiento de la ventana k ≥ 1 depende de los trabajos base que
    eligen su padre en k-1 y no existe en la primera ventana. Un trabajo de RL depende de
    la selección de la cadena de cada ventana que lee (ajuste, validación y evaluación)
    con la semilla del predictor que declara en `predictor_seed`. Su `predictor` puede
    nombrar el brazo base o el brazo de la cadena.
    """
    for job in stages.get("adapters", ()):
        parent = parent_window(campaign, job["scope"], job["window"])
        _require(parent is not None, f"{job['id']}: la primera ventana no tiene posentrenamiento")
        needed = parent_jobs(base_jobs, job["scope"], parent, job["base_arm"], job["seed"])
        _require(
            set(needed) <= set(job["depends"]),
            f"{job['id']} no depende del estado elegido de la base en {parent}",
        )
    for job in stages.get("rl", ()):
        windows = [*job["train"], job["validation"], job["window"]]
        seed = job.get("predictor_seed")
        _require(type(seed) is int, f"{job['id']} no declara la semilla de su predictor")
        arm = job["predictor"].removesuffix(CHAIN_SUFFIX)
        needed = {chain_job_id(job["scope"], window, arm, seed) for window in windows}
        _require(
            len(job["train"]) >= MIN_RL_TRAIN_WINDOWS and needed <= set(job["depends"]),
            f"{job['id']} no depende de la cadena de todas las ventanas que lee",
        )


def asset_digest(sample_rows):
    """Número de filas y huella de las filas de un activo, independiente de su orden."""
    rows = np.sort(np.asarray(sample_rows))
    _require(
        rows.ndim == 1 and rows.dtype.kind in "iu" and not (rows[1:] == rows[:-1]).any(),
        "Las filas de un activo son enteros sin repetir",
    )
    return len(rows), hashlib.sha256(rows.astype("<i8").tobytes()).hexdigest()


def combine(parts):
    """Huella de un conjunto de filas a partir de las de sus activos con filas.

    `parts` asigna a cada (mercado, activo) su resultado de `asset_digest`.
    """
    digest, total = hashlib.sha256(FINGERPRINT), 0
    for (market, symbol), (rows, value) in sorted(parts.items()):
        _require(rows > 0, f"{market}/{symbol} no aporta filas a la huella")
        digest.update(f"{market}/{symbol}:{rows}:{value}\n".encode())
        total += rows
    return total, digest.hexdigest()


def row_fingerprint(markets, symbols, sample_rows):
    """Número de filas y huella de filas (mercado, activo, fila de la muestra).

    Es la de `combine` sobre `asset_digest` de cada activo, así que se puede calcular de una
    vez o activo a activo con el mismo resultado.
    """
    markets, symbols = np.asarray(markets, dtype=str), np.asarray(symbols, dtype=str)
    rows = np.asarray(sample_rows)
    _require(
        markets.shape == symbols.shape == rows.shape and rows.ndim == 1,
        "Cada fila necesita mercado, activo y fila de la muestra",
    )
    parts = {}
    for market, symbol in sorted(set(zip(markets.tolist(), symbols.tolist(), strict=True))):
        parts[market, symbol] = asset_digest(rows[(markets == market) & (symbols == symbol)])
    return combine(parts)


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
    selected, first = document["selected"], document["parent_window"] is None
    if first:
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
