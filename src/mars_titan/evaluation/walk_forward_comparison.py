"""Comparar brazos de un walk-forward anual con las mismas filas y sin abrir 2024.

La configuración se declara antes de evaluar: política de entradas, protocolos,
ventanas, brazos, métricas, calibración común y familias de contrastes. Un
manifiesto de fuentes enlaza cada brazo, semilla y ventana con sus predicciones
del tramo de evaluación y, si emite cuantiles, con las del tramo de calibración.

Cada ventana se puntúa por separado con ``score_sessions`` y las sesiones se unen
después. Así ninguna ventana supera el presupuesto de filas del panel y las
métricas agregadas coinciden con las de un panel único. Las comparaciones
promedian semillas sesión a sesión y usan ``compare_series``.

Se rechaza cualquier mezcla: otra política, otra vista o edición, otro protocolo,
ventanas ausentes o sobrantes, filas fuera del tramo declarado, filas de la
reserva de 2024 y brazos que no evalúan exactamente las mismas filas (activo,
mercado e instante) con los mismos objetivos. El mensaje indica cuántas difieren.

La versión 2 de la configuración declara además los estratos por presencia de
noticias y fundamentales (``modality_strata``), un análisis secundario que no
cambia ninguna salida de la versión 1. Su presencia sale de la propia vista de
cada ventana y debe cubrir exactamente las filas evaluadas por los brazos.

La versión 3 añade la ablación de modalidades en inferencia (``modality_ablation``),
otro análisis secundario. Sus predicciones enmascaradas llegan en un manifiesto propio
de la etapa de ablación y se comparan con las originales en las mismas filas, con el
calibrador ya ajustado. Sin ese manifiesto, la sección queda pendiente y el resto del
informe no cambia.

La versión 4 declara la cartera larga y corta por cuartiles (``long_short``), que calcula
``long_short_comparison`` con estas mismas fuentes y comprobaciones. El informe añade en
todas las versiones la fiabilidad de la probabilidad implícita de subida
(``sign_reliability``): ECE medio de las semillas con intervalo percentil por bloques de
días y curva de fiabilidad, en bruto y con el calibrador común.

La versión 5 añade el diseño conjunto (``joint_design``). El ámbito conjunto compara todos
los brazos y un mercado solo cuenta en las ventanas en las que su propio protocolo, con su
historia mínima, recorre los mismos tramos (``market_eligibility``). Las filas de ese
mercado en las demás ventanas se predicen y se conservan, pero quedan fuera de la
calibración y de las métricas, y el informe cuenta cuántas se excluyen. Cada ámbito de un
solo mercado compara los controles separados con el mismo brazo conjunto restringido a
las filas de su mercado (``<brazo><sufijo>``), en la ventana con los mismos tramos, con una
familia de diferencias conjunto menos separado. Sin esta sección cada ámbito compara todos
los brazos con todas sus filas, como antes.
"""

import argparse
import dataclasses
import json
import resource
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.calibration import conformal_quantiles as cqr
from mars_titan.data import prediction_files
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import masked_inputs, policy_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import long_short, modality_ablation, modality_strata
from mars_titan.evaluation.forecast_panel import WEIGHTINGS, ForecastPanel, SessionSeries
from mars_titan.evaluation.forecast_scores import (
    COVERAGE_ERROR,
    INTERVAL_SCORE,
    QUANTILE_SERIES,
    SIGN_BINS,
    SessionScores,
    expected_calibration_error,
    score_sessions,
)
from mars_titan.evaluation.paired_comparisons import (
    circular_block_counts,
    compare_series,
    delta,
    interaction,
    level,
)
from mars_titan.evaluation.splits import build_folds, eligible_folds
from mars_titan.models.quantile_head import LEVELS, QUANTILE_COLUMNS, QUANTILE_HEAD
from mars_titan.training.temporal_contract import temporal_contracts

CONFIG_KIND = "walk_forward_comparison"
SOURCES_KIND = "walk_forward_prediction_sources"
REPORT_KIND = "walk_forward_comparison_report"
DECLARED = "declared_before_evaluation"
FINAL_TEST_START = "2024-01-01"
SCOPES = {"US": ("US",), "CN": ("CN",), "US+CN": ("US", "CN")}
ZERO_CONTROL = "zero_control"
POINT = "point"
OUTPUTS = (ZERO_CONTROL, POINT, QUANTILE_HEAD)
SERIES_METRICS = (
    "mae",
    "mse",
    "direction_accuracy",
    "rank_ic",
    "pinball",
    "up_precision",
    "down_precision",
    "sign_brier",
    f"{INTERVAL_SCORE}0.8",
    f"{INTERVAL_SCORE}0.95",
)
COLUMNS = ("asset_id", "market", "prediction_at", "target", "prediction")
MAX_FILE_BYTES = 4 * 1024**3
MAX_ARMS = 64
ABSTENTION = dict(
    point_rule="zero_point_prediction_is_abstention",
    point_metrics=["direction_accuracy", "conditional_direction_accuracy", "call_coverage"],
    interval_rule="interval_excluding_zero_commits_to_a_sign",
    interval_metrics=["sign_commitment", "committed_sign_error"],
    threshold_rule=None,
    threshold_reason=(
        "No hay una regla de abstención con umbral declarada antes de evaluar. No se crea aquí."
    ),
)
_CONFIG_FIELDS = {
    "schema_version",
    "kind",
    "name",
    "status",
    "input_policy",
    "partition",
    "scopes",
    "arms",
    "metrics",
    "calibration",
    "comparison",
}
STRATA_FIELD = "modality_strata"
ABLATION_FIELD = "modality_ablation"
LONG_SHORT_FIELD = "long_short"
JOINT_FIELD = "joint_design"
# Secciones secundarias que añade cada versión de la configuración.
SECTIONS = {
    1: set(),
    2: {STRATA_FIELD},
    3: {STRATA_FIELD, ABLATION_FIELD},
    4: {STRATA_FIELD, ABLATION_FIELD, LONG_SHORT_FIELD},
    5: {STRATA_FIELD, ABLATION_FIELD, LONG_SHORT_FIELD, JOINT_FIELD},
}
_JOINT_FIELDS = {
    "declared_at",
    "joint_scope",
    "market_eligibility",
    "ineligible_rows",
    "separate_controls",
    "joint_suffix",
    "contrast_family",
    "seeds",
}
INELIGIBLE_ROWS = "predicted_and_kept_excluded_from_calibration_and_metrics"
SEED_AGGREGATION = "summaries_per_seed_and_session_mean_series_for_contrasts"
_METRIC_FIELDS = {"primary", "market_weighting", "rank_ic_min_assets", "quantile_head"}
_CALIBRATION_FIELDS = {"method", "partition", "nominals", "groups", "min_rows", "order_rule"}
_COMPARISON_FIELDS = {
    "metrics",
    "block_length",
    "sensitivity_block_lengths",
    "replicates",
    "seed",
    "confidence",
    "families",
}
_FAMILY_FIELDS = {
    "delta": {"kind", "base", "variants"},
    "factorial": {"kind", "base", "first", "second", "joint"},
    "level": {"kind", "arms"},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _microseconds(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def _name(value):
    return isinstance(value, str) and 0 < len(value) <= 80 and value.replace("_", "").isalnum()


def _protocols(folder, scope, declared):
    """Leer los protocolos de un ámbito y exigir cortes comunes en el conjunto."""
    markets = SCOPES[scope]
    _require(
        isinstance(declared, dict) and set(declared) == {"protocols", "windows"},
        f"El ámbito {scope} declara protocolos y ventanas",
    )
    paths = declared["protocols"]
    _require(
        isinstance(paths, dict) and set(paths) == set(markets),
        f"El ámbito {scope} necesita un protocolo por mercado",
    )
    protocols, digests = {}, {}
    for market in markets:
        path = Path(paths[market])
        path = path if path.is_absolute() else folder / path
        protocol, digest = read_manifest(path, 1024**2)
        _require(
            isinstance(protocol, dict)
            and protocol.get("market") == market
            and protocol.get("schema_version") == 2
            and protocol.get("final_test_start") == FINAL_TEST_START,
            f"El protocolo {market} de {scope} no es v2 o mueve la reserva final",
        )
        protocols[market], digests[market] = protocol, digest
    first = protocols[markets[0]]
    _require(
        all(
            {k: v for k, v in protocols[market].items() if k != "market"}
            == {k: v for k, v in first.items() if k != "market"}
            for market in markets
        ),
        f"Los mercados de {scope} no comparten cortes",
    )
    folds = {fold["id"]: fold for fold in build_folds(first)}
    windows = declared["windows"]
    if windows == "all":
        windows = list(folds)
    _require(
        isinstance(windows, list)
        and windows
        and len(set(windows)) == len(windows)
        and all(window in folds for window in windows)
        and windows == sorted(windows),
        f"Las ventanas de {scope} deben ser 'all' o ventanas del protocolo en orden",
    )
    _require(
        all(folds[window]["evaluation"][1] <= FINAL_TEST_START for window in windows),
        "Una ventana invade la reserva final",
    )
    return dict(
        markets=list(markets),
        protocols=protocols,
        protocol_sha256=digests,
        windows={window: folds[window] for window in windows},
    )


def _distinct(arms, name):
    _require(
        isinstance(arms, list) and arms and len(set(arms)) == len(arms),
        f"La familia {name} necesita brazos distintos",
    )
    return arms


def _families(families, arms):
    """Traducir cada familia declarada a los coeficientes de ``compare_series``."""
    _require(isinstance(families, dict) and families, "Faltan familias de contrastes")
    resolved = {}
    for name, family in families.items():
        kind = family.get("kind") if isinstance(family, dict) else None
        _require(
            _name(name) and kind in _FAMILY_FIELDS and set(family) == _FAMILY_FIELDS[kind],
            f"La familia {name} no está bien declarada",
        )
        if kind == "delta":
            base = family["base"]
            variants = _distinct([base, *family["variants"]], name)[1:]
            contrasts = {f"{variant}-{base}": delta(base, variant) for variant in variants}
        elif kind == "factorial":
            base, first, second, joint = _distinct(
                [family[key] for key in ("base", "first", "second", "joint")], name
            )
            contrasts = {
                f"{first}-{base}": delta(base, first),
                f"{second}-{base}": delta(base, second),
                f"{joint}-{base}": delta(base, joint),
                "interaction": interaction(base, first, second, joint),
            }
        else:
            contrasts = {arm: level(arm) for arm in _distinct(family["arms"], name)}
        used = {arm for coefficients in contrasts.values() for arm in coefficients}
        _require(used <= set(arms), f"La familia {name} usa brazos no declarados")
        resolved[name] = contrasts
    return resolved


def _plain_scopes(resolved, arms, families):
    """Sin diseño conjunto: cada ámbito compara todos los brazos con todas sus filas."""
    for scope in resolved.values():
        scope.update(
            arms=dict(arms),
            families=dict(families),
            eligible={market: list(scope["windows"]) for market in scope["markets"]},
            borrowed={},
            joint_windows={},
        )


def _joint_design(declared, resolved, arms, families, folder):
    """Ámbito conjunto con elegibilidad por mercado y controles separados por mercado.

    Cada ventana de un ámbito de un mercado se empareja con la ventana conjunta que tiene
    los mismos cuatro tramos y en la que ese mercado es elegible.
    """
    _require(
        isinstance(declared, dict)
        and set(declared) == _JOINT_FIELDS
        and isinstance(declared["declared_at"], str)
        and declared["ineligible_rows"] == INELIGIBLE_ROWS
        and declared["seeds"] == SEED_AGGREGATION
        and _name(declared["contrast_family"])
        and declared["contrast_family"] not in families
        and isinstance(declared["joint_suffix"], str)
        and _name(f"x{declared['joint_suffix']}"),
        "El diseño conjunto no cumple su contrato",
    )
    joint_name = declared["joint_scope"]
    joint = resolved.get(joint_name)
    _require(
        joint is not None and len(joint["markets"]) > 1,
        "El diseño conjunto necesita un ámbito con varios mercados",
    )
    _plain_scopes({joint_name: joint}, arms, families)
    eligibility = declared["market_eligibility"]
    _require(
        isinstance(eligibility, dict) and eligibility and set(eligibility) <= set(joint["markets"]),
        "La elegibilidad se declara para mercados del ámbito conjunto",
    )
    references = {}
    for market, value in eligibility.items():
        path = Path(value)
        reference, digest = read_manifest(path if path.is_absolute() else folder / path, 1024**2)
        allowed = eligible_folds(joint["protocols"][market], reference)
        joint["eligible"][market] = [window for window in joint["windows"] if window in allowed]
        references[market] = digest
    joint["eligibility_sha256"] = references
    controls = declared["separate_controls"]
    suffix = declared["joint_suffix"]
    _require(
        isinstance(controls, list)
        and controls
        and len(set(controls)) == len(controls)
        and all(arms.get(arm, {}).get("output") not in (None, ZERO_CONTROL) for arm in controls)
        and not {f"{arm}{suffix}" for arm in controls} & set(arms)
        and all(_name(f"{arm}{suffix}") for arm in controls),
        "Los controles separados deben ser brazos con predicciones y su alias, un nombre libre",
    )
    zero = {name: arm for name, arm in arms.items() if arm["output"] == ZERO_CONTROL}
    for name, scope in resolved.items():
        if name == joint_name:
            continue
        (market,) = _require_single(scope, name, joint)
        folds = {
            window: tuple(
                tuple(fold[part]) for part in ("train", "validation", "calibration", "evaluation")
            )
            for window, fold in joint["windows"].items()
        }
        pairs = {}
        for window, fold in scope["windows"].items():
            key = tuple(
                tuple(fold[part]) for part in ("train", "validation", "calibration", "evaluation")
            )
            matches = [other for other, value in folds.items() if value == key]
            _require(
                len(matches) == 1 and matches[0] in joint["eligible"][market],
                f"La ventana {window} de {name} no tiene una ventana conjunta elegible con los "
                "mismos tramos",
            )
            pairs[window] = matches[0]
        aliases = {f"{arm}{suffix}": arm for arm in controls}
        scope.update(
            arms={
                **zero,
                **{arm: arms[arm] for arm in controls},
                **{alias: arms[arm] for alias, arm in aliases.items()},
            },
            families={
                declared["contrast_family"]: {
                    f"{alias}-{arm}": delta(arm, alias) for alias, arm in aliases.items()
                }
            },
            eligible={market: list(scope["windows"])},
            borrowed=dict(aliases),
            joint_windows=pairs,
            joint_scope=joint_name,
        )


def _require_single(scope, name, joint):
    markets = scope["markets"]
    _require(
        len(markets) == 1 and markets[0] in joint["markets"],
        f"El ámbito {name} debe ser de un solo mercado del ámbito conjunto",
    )
    return markets


def load_config(path):
    """Validar la configuración declarada antes de abrir ninguna predicción."""
    path = Path(path)
    config, digest = read_manifest(path, 1024**2)
    return validate_config(config, digest, path.parent)


def resolve_config(config):
    """Ruta de una configuración declarada o configuración ya validada por ``validate_config``.

    Una configuración derivada (por ejemplo, la de los brazos postentrenados de un padre)
    llega ya validada, con su huella y sus ámbitos resueltos.
    """
    if isinstance(config, dict):
        _require(
            {"sha256", "resolved_scopes", "resolved_families"} <= set(config),
            "La configuración en memoria debe llegar validada",
        )
        return config
    return load_config(config)


def validate_config(config, digest, folder):
    """Validar una configuración ya leída. ``folder`` resuelve las rutas de los protocolos.

    Cada versión añade las secciones secundarias de ``SECTIONS``. La 4 añade la cartera
    larga y corta por cuartiles y la 5 el diseño conjunto con controles separados.
    """
    version = config.get("schema_version") if isinstance(config, dict) else None
    _require(
        isinstance(config, dict)
        and type(version) is int
        and version in SECTIONS
        and set(config) == _CONFIG_FIELDS | SECTIONS[version]
        and config["kind"] == CONFIG_KIND
        and config["status"] == DECLARED
        and config["partition"] == "evaluation"
        and isinstance(config["name"], str)
        and _name(config["name"].replace("-", "_")),
        "La configuración no cumple su contrato",
    )
    masked_inputs(config["input_policy"])
    scopes = config["scopes"]
    _require(
        isinstance(scopes, dict) and scopes and set(scopes) <= set(SCOPES),
        "Los ámbitos deben ser US, CN o US+CN",
    )
    arms = config["arms"]
    _require(
        isinstance(arms, dict) and 2 <= len(arms) <= MAX_ARMS and all(map(_name, arms)),
        "Los brazos deben tener nombres válidos y estar acotados",
    )
    for name, arm in arms.items():
        _require(
            isinstance(arm, dict)
            and set(arm) == {"family", "output", "seeds"}
            and _name(arm["family"])
            and arm["output"] in OUTPUTS
            and isinstance(arm["seeds"], list)
            and len(set(arm["seeds"])) == len(arm["seeds"])
            and all(type(seed) is int and 0 <= seed < 2**32 for seed in arm["seeds"])
            and (arm["seeds"] == []) == (arm["output"] == ZERO_CONTROL),
            f"El brazo {name} no declara familia, salida y semillas válidas",
        )
    _require(
        any(arm["output"] != ZERO_CONTROL for arm in arms.values()),
        "Se necesita al menos un brazo con predicciones",
    )
    metrics = config["metrics"]
    _require(
        isinstance(metrics, dict)
        and set(metrics) == _METRIC_FIELDS
        and metrics["primary"] == "mae"
        and metrics["market_weighting"] in WEIGHTINGS
        and type(metrics["rank_ic_min_assets"]) is int
        and metrics["rank_ic_min_assets"] >= 3
        and metrics["quantile_head"] == QUANTILE_HEAD,
        "Las métricas declaradas no son válidas",
    )
    calibration = config["calibration"]
    _require(
        isinstance(calibration, dict)
        and set(calibration) == _CALIBRATION_FIELDS
        and calibration["method"] == cqr.METHOD
        and calibration["partition"] == cqr.PARTITION
        and calibration["groups"] == "market"
        and calibration["order_rule"] == cqr.ORDER_RULE
        and type(calibration["min_rows"]) is int
        and calibration["min_rows"] >= 1,
        "La calibración declarada no es válida",
    )
    cqr.interval_pairs(LEVELS, calibration["nominals"])
    comparison = config["comparison"]
    _require(
        isinstance(comparison, dict)
        and set(comparison) == _COMPARISON_FIELDS
        and isinstance(comparison["metrics"], list)
        and comparison["metrics"]
        and comparison["metrics"][0] == "mae"
        and set(comparison["metrics"]) <= set(SERIES_METRICS)
        and len(set(comparison["metrics"])) == len(comparison["metrics"]),
        "Las comparaciones deben empezar por el MAE y usar métricas por sesión",
    )
    families = _families(comparison["families"], arms)
    if version >= 2:
        _require(
            masked_inputs(config["input_policy"]),
            "Los estratos de presencia necesitan la política de entradas con máscaras",
        )
        modality_strata.declaration(config[STRATA_FIELD], SERIES_METRICS)
    if version >= 3:
        modality_ablation.declaration(config[ABLATION_FIELD])
    if version >= 4:
        long_short.declaration(config[LONG_SHORT_FIELD])
    resolved = {scope: _protocols(folder, scope, declared) for scope, declared in scopes.items()}
    if JOINT_FIELD in config:
        _joint_design(config[JOINT_FIELD], resolved, arms, families, folder)
    else:
        _plain_scopes(resolved, arms, families)
    return dict(config, sha256=digest, resolved_scopes=resolved, resolved_families=families)


def scope_config(config, scope):
    """Configuración vista desde un ámbito: sus brazos y sus familias de contrastes."""
    resolved = config["resolved_scopes"][scope]
    return dict(config, arms=resolved["arms"], resolved_families=resolved["families"])


def _file(folder, record, label, *, predictions=False):
    """Ruta y huella de una fuente. Unas predicciones pueden estar compactadas o liberadas."""
    _require(
        isinstance(record, dict)
        and set(record) == {"path", "sha256"}
        and isinstance(record["path"], str)
        and record["path"]
        and isinstance(record["sha256"], str)
        and len(record["sha256"]) == 64,
        f"{label} necesita ruta y huella",
    )
    path = Path(record["path"])
    path = path if path.is_absolute() else folder / path
    if predictions and not path.exists() and prediction_files.entry(path) is not None:
        # La retención v2 sustituye las filas por su forma compacta o por sus huellas. Un
        # archivo que falta sin ese registro se rechaza abajo como cualquier otra fuente.
        prediction_files.verify(path, record["sha256"])
        return dict(path=path, sha256=record["sha256"])
    _require(not path.is_symlink() and path.is_file(), f"{label} no es un archivo regular")
    _require(0 < path.stat().st_size <= MAX_FILE_BYTES, f"{label} supera el presupuesto")
    return dict(path=path, sha256=record["sha256"])


def _view(folder, record, window, scope, policy):
    """Validar la vista de una ventana con la política declarada y devolver su edición."""
    source = _file(folder, record, f"La vista de {window['id']}")
    manifest, digest = read_manifest(source["path"], 64 * 1024**2)
    _require(digest == source["sha256"], f"La vista de {window['id']} cambió su huella")
    _require(
        manifest.get("kind") == "corpus_supervision" and manifest.get("final_test_opened") is False,
        f"La vista de {window['id']} no es una supervisión con la reserva cerrada",
    )
    contracts = temporal_contracts(manifest, input_policy=policy)
    _require(
        set(contracts) == set(scope["markets"]),
        f"La vista de {window['id']} no declara los mercados del ámbito",
    )
    edition = {}
    for market, contract in contracts.items():
        _require(
            contract["protocol"] == scope["protocols"][market] and contract["fold"] == window,
            f"La vista de {window['id']} no conserva el protocolo o la ventana declarados",
        )
        edition[market] = contract["parent_sha256"]
    return digest, edition


def load_sources(path, config, scope_name):
    """Validar el manifiesto de fuentes de un ámbito antes de leer predicciones."""
    path = Path(path)
    sources, digest = read_manifest(path, 16 * 1024**2)
    scope = config["resolved_scopes"].get(scope_name)
    _require(scope is not None, "El ámbito no está declarado en la configuración")
    _require(
        isinstance(sources, dict)
        and set(sources) == {"schema_version", "kind", "scope", "input_policy", "windows", "arms"}
        and sources["schema_version"] == 1
        and sources["kind"] == SOURCES_KIND
        and sources["scope"] == scope_name,
        "Las fuentes no cumplen su contrato",
    )
    policy = config["input_policy"]
    _require(
        sources["input_policy"] == policy,
        "Las fuentes declaran otra política de entradas que la configuración",
    )
    windows = scope["windows"]
    _require(
        isinstance(sources["windows"], dict) and set(sources["windows"]) == set(windows),
        "Las fuentes no cubren exactamente las ventanas declaradas",
    )
    # Un ámbito con brazos prestados del conjunto declara también la vista conjunta emparejada.
    borrowed = scope["borrowed"]
    fields = {"view", "joint_view"} if borrowed else {"view"}
    views, view_paths, edition, joint_views = {}, {}, None, {}
    for window_id, window in windows.items():
        entry = sources["windows"][window_id]
        _require(isinstance(entry, dict) and set(entry) == fields, "La ventana necesita su vista")
        views[window_id], current = _view(path.parent, entry["view"], window, scope, policy)
        if borrowed:
            joint = config["resolved_scopes"][scope["joint_scope"]]
            pair = scope["joint_windows"][window_id]
            joint_views[window_id], joint_edition = _view(
                path.parent, entry["joint_view"], joint["windows"][pair], joint, policy
            )
            _require(
                all(joint_edition[market] == current[market] for market in current),
                f"La vista conjunta de {window_id} parte de otra edición",
            )
        # La ruta solo se abre si se declaran estratos, y entonces se comprueba su huella.
        view_paths[window_id] = _file(path.parent, entry["view"], f"La vista de {window_id}")[
            "path"
        ]
        _require(
            edition is None or current == edition, "Las ventanas no comparten la misma edición"
        )
        edition = current
    declared = {name: arm for name, arm in scope["arms"].items() if arm["output"] != ZERO_CONTROL}
    arms = sources["arms"]
    _require(
        isinstance(arms, dict) and set(arms) == set(declared),
        "Las fuentes no contienen exactamente los brazos declarados con predicciones",
    )
    files = {}
    for name, arm in declared.items():
        seeds = arms[name]
        expected = {str(seed) for seed in arm["seeds"]}
        _require(
            isinstance(seeds, dict) and set(seeds) == expected,
            f"El brazo {name} no contiene exactamente sus semillas declaradas",
        )
        quantile = arm["output"] == QUANTILE_HEAD
        for seed, entries in seeds.items():
            label = f"El brazo {name} semilla {seed}"
            present = set(entries) if isinstance(entries, dict) else set()
            _require(
                present == set(windows),
                f"{label} no cubre las ventanas declaradas: faltan "
                f"{sorted(set(windows) - present)} y sobran {sorted(present - set(windows))}",
            )
            for window_id, entry in entries.items():
                where = f"{label} en {window_id}"
                fields = {"input_policy", "view_sha256", "evaluation"}
                _require(
                    isinstance(entry, dict)
                    and set(entry) == fields | ({"calibration"} if quantile else set()),
                    f"{where} no declara política, vista y predicciones de su salida",
                )
                _require(entry["input_policy"] == policy, f"{where} declara otra política")
                expected = (joint_views if name in borrowed else views)[window_id]
                _require(entry["view_sha256"] == expected, f"{where} usa otra vista")
                files[name, int(seed), window_id] = {
                    part: _file(path.parent, entry[part], f"{where} ({part})", predictions=True)
                    for part in ("calibration", "evaluation")
                    if part in entry
                }
    return dict(
        sha256=digest,
        scope=scope_name,
        views=views,
        view_paths=view_paths,
        edition=edition,
        files=files,
        joint_views=joint_views,
        **scope,
    )


def restrict_windows(config, scope, windows):
    """La configuración validada con un ámbito limitado a algunas de sus ventanas.

    Conserva la huella de la configuración, porque declara lo mismo. Sirve para validar y
    puntuar las fuentes de una ventana en cuanto termina, antes de tener las demás.
    """
    resolved = config["resolved_scopes"][scope]
    windows = set(windows)
    _require(
        windows and windows <= set(resolved["windows"]),
        "Las ventanas deben ser del ámbito declarado",
    )
    kept = {key: value for key, value in resolved["windows"].items() if key in windows}
    narrowed = dict(resolved, windows=kept)
    # La elegibilidad por mercado y el emparejamiento con el conjunto siguen a las ventanas.
    if "eligible" in resolved:
        narrowed["eligible"] = {
            market: [name for name in names if name in windows]
            for market, names in resolved["eligible"].items()
        }
    if "joint_windows" in resolved:
        narrowed["joint_windows"] = {
            name: pair for name, pair in resolved["joint_windows"].items() if name in windows
        }
    scopes = dict(config["resolved_scopes"], **{scope: narrowed})
    return dict(config, resolved_scopes=scopes)


def _read_predictions(file, columns):
    """Leer solo las columnas necesarias después de comprobar huella y tipos.

    Un archivo compactado por la retención v2 se lee igual que el original. Uno liberado
    no tiene filas: su ventana se compara con los agregados guardados o tras regenerarlo.
    """
    table = prediction_files.read(file["path"], file["sha256"], columns)
    _require(
        table.schema.field("prediction_at").type == pa.timestamp("us", tz="UTC"),
        "Los instantes deben ser timestamp UTC en microsegundos",
    )
    _require(all(table[name].null_count == 0 for name in columns), "Hay valores ausentes")
    for name in ("asset_id", "market"):
        _require(
            pa.types.is_string(table[name].type) or pa.types.is_large_string(table[name].type),
            f"La columna {name} debe ser texto",
        )
    return table


def _check_segment(table, window, partition, markets, label):
    """Exigir el tramo declarado de la ventana para cada mercado, sin filtrar en silencio."""
    times = table["prediction_at"].cast(pa.int64()).to_numpy()
    market = table["market"].to_numpy(zero_copy_only=False)
    reserved = int(np.count_nonzero(times >= _microseconds(FINAL_TEST_START)))
    _require(reserved == 0, f"{label}: {reserved} filas pertenecen a la reserva final de 2024")
    unknown = int(np.count_nonzero(~np.isin(market, markets)))
    _require(unknown == 0, f"{label}: {unknown} filas de mercados fuera del ámbito")
    start, end = (_microseconds(day) for day in window[partition])
    outside = int(np.count_nonzero((times < start) | (times >= end)))
    _require(
        outside == 0,
        f"{label}: {outside} filas fuera del tramo de {partition} {window[partition]}",
    )
    return market, times


def _restricted(table, sources, window_id, arm=None):
    """Quitar las filas que el diseño excluye de la ventana y contarlas por mercado.

    En el ámbito conjunto salen las filas de los mercados no elegibles en la ventana. Un
    brazo prestado del conjunto conserva solo las filas del mercado del ámbito. Cualquier
    otro mercado sigue llegando a la comprobación del tramo, que lo rechaza.
    """
    excluded, counts = _excluded(table, sources, window_id, arm)
    return (table, counts) if excluded is None else (table.filter(pa.array(~excluded)), counts)


def _excluded(table, sources, window_id, arm=None):
    """Máscara de las filas que el diseño excluye y su recuento por mercado."""
    drop = [market for market in sources["markets"] if window_id not in sources["eligible"][market]]
    if arm in sources["borrowed"]:
        joint = sources["joint_scope"]
        drop += [m for m in SCOPES[joint] if m not in sources["markets"]]
    if not drop:
        return None, {}
    market = table["market"].to_numpy(zero_copy_only=False)
    excluded = np.isin(market, drop)
    return excluded, {name: int(np.count_nonzero(market == name)) for name in drop}


def _row_id(table):
    """Identidad de fila (mercado, activo, instante), común a todos los runners."""
    return pc.binary_join_element_wise(
        table["market"].cast(pa.large_string()),
        table["asset_id"].cast(pa.large_string()),
        table["prediction_at"].cast(pa.int64()).cast(pa.large_string()),
        pa.scalar("/", pa.large_string()),
    )


def _panel(table, window, partition, scope, label, *, output, prediction=None):
    market, times = _check_segment(table, window, partition, scope["markets"], label)
    quantile = output == QUANTILE_HEAD
    return ForecastPanel.from_columns(
        _row_id(table),
        market,
        times,
        table["target"].to_numpy(),
        table["prediction"].to_numpy() if prediction is None else prediction,
        quantiles=(
            np.column_stack([table[name].to_numpy() for name in QUANTILE_COLUMNS])
            if quantile
            else None
        ),
        levels=LEVELS if quantile else None,
        markets=tuple(scope["markets"]),
    )


def _same_rows(reference, panel, label):
    """Exigir las mismas filas y objetivos que el primer brazo, contando las diferencias."""
    reference_label, expected = reference
    if expected.cohort_sha256 == panel.cohort_sha256:
        return
    only_here = pc.sum(pc.invert(pc.is_in(panel.row_id, value_set=expected.row_id))).as_py()
    only_there = pc.sum(pc.invert(pc.is_in(expected.row_id, value_set=panel.row_id))).as_py()
    changed = 0
    if only_here == only_there == 0:
        changed = int(np.count_nonzero(expected.target != panel.target))
    raise ValueError(
        f"{label} no evalúa las mismas filas que {reference_label}: {only_here} filas solo "
        f"en este brazo, {only_there} filas solo en la referencia y {changed} filas con "
        "otro objetivo"
    )


def _calibrated(panel, record):
    """Panel con los cuantiles calibrados. La predicción puntual y los objetivos no cambian."""
    groups = np.asarray(panel.markets)[panel.market]
    missing = cqr.undefined_groups(record, groups)
    if missing:
        return None, None, f"Sin corrección definida para {missing}"
    quantiles, adjusted = cqr.apply_conformal_quantiles(record, panel.quantiles, groups)
    quantiles.setflags(write=False)
    return dataclasses.replace(panel, quantiles=quantiles), adjusted, None


def _presence_bits(sources, config, window_id, reference):
    """Presencia de cada fila evaluada, en el orden canónico del panel de referencia.

    La presencia se lee de la vista de la ventana y debe cubrir las mismas filas con
    los mismos objetivos que los brazos, con la misma comprobación que entre brazos.
    El registro cuenta las filas sin precios, gráficos o macro. Con alguna, los estratos
    declarados no describen la ventana.
    """
    table, bits = modality_strata.view_presence(
        sources["view_paths"][window_id], config["input_policy"], sources["views"][window_id]
    )
    excluded = _excluded(table, sources, window_id)[0]
    if excluded is not None:
        table, bits = table.filter(pa.array(~excluded)), bits[~excluded]
    label = f"La presencia de la vista de {window_id}"
    panel = _panel(
        table,
        sources["windows"][window_id],
        "evaluation",
        sources,
        label,
        output=POINT,
        prediction=np.zeros(table.num_rows),
    )
    _same_rows(reference, panel, label)
    record = dict(rows=table.num_rows, incomplete=modality_strata.incomplete_rows(bits))
    order = pc.index_in(reference[1].row_id, value_set=_row_id(table)).to_numpy()
    return bits[order], record


def _ablation_sources(path, config, sources):
    """Validar el manifiesto de predicciones enmascaradas frente a las fuentes principales.

    Cada variante declara, para cada brazo con predicciones, semilla y ventana, la
    evaluación que predijo el estado elegido con la modalidad ausente. Las vistas y la
    configuración deben ser las de la comparación.
    """
    path = Path(path)
    document, digest = read_manifest(path, 16 * 1024**2)
    fields = {"schema_version", "kind", "scope", "input_policy", "comparison_sha256", "windows"}
    _require(
        isinstance(document, dict)
        and set(document) == fields | {"variants"}
        and document["schema_version"] == 1
        and document["kind"] == modality_ablation.SOURCES_KIND
        and document["scope"] == sources["scope"],
        "Las fuentes de la ablación no cumplen su contrato",
    )
    _require(
        document["input_policy"] == config["input_policy"]
        and document["comparison_sha256"] == config["sha256"]
        and document["windows"] == sources["views"],
        "Las fuentes de la ablación declaran otra política, comparación o vista",
    )
    variants = document["variants"]
    _require(
        isinstance(variants, dict) and set(variants) == set(modality_ablation.VARIANTS),
        "Las fuentes de la ablación no contienen exactamente las variantes declaradas",
    )
    declared = {name: arm for name, arm in config["arms"].items() if arm["output"] != ZERO_CONTROL}
    files = {}
    for variant, arms in variants.items():
        _require(
            isinstance(arms, dict) and set(arms) == set(declared),
            f"{variant} no contiene exactamente los brazos declarados con predicciones",
        )
        for name, arm in declared.items():
            seeds = arms[name]
            _require(
                isinstance(seeds, dict) and set(seeds) == {str(seed) for seed in arm["seeds"]},
                f"{variant} no contiene exactamente las semillas de {name}",
            )
            for seed, entries in seeds.items():
                where = f"{variant} de {name} semilla {seed}"
                _require(
                    isinstance(entries, dict) and set(entries) == set(sources["windows"]),
                    f"{where} no cubre exactamente las ventanas declaradas",
                )
                for window_id, entry in entries.items():
                    _require(
                        isinstance(entry, dict) and set(entry) == {"evaluation"},
                        f"{where} en {window_id} solo declara su evaluación",
                    )
                    files[variant, name, int(seed), window_id] = _file(
                        path.parent,
                        entry["evaluation"],
                        f"{where} en {window_id}",
                        predictions=True,
                    )
    return dict(sha256=digest, files=files)


def _masked_scores(ablation, sources, window_id, arm, seed, original, calibrated, record, context):
    """Puntuaciones de las variantes de un brazo y semilla frente a su predicción original.

    Las predicciones enmascaradas pasan las mismas comprobaciones de tramo y filas que las
    originales y se corrigen con el mismo calibrador, sin volver a ajustarlo.
    """
    name, output, columns, bits, reference, minimum = context
    result = {}
    for variant in modality_ablation.VARIANTS:
        label = f"{name} semilla {seed} en {window_id} con {variant}"
        table = _read_predictions(ablation["files"][variant, name, seed, window_id], columns)
        table = _restricted(table, sources, window_id, name)[0]
        window = sources["windows"][window_id]
        masked = _panel(table, window, "evaluation", sources, label, output=output)
        _same_rows(reference, masked, label)
        adjusted = None if record is None else _calibrated(masked, record)[0]
        result[variant] = modality_ablation.score_pair(
            original,
            masked,
            modality_ablation.affected_rows(bits, variant),
            rank_ic_min_assets=minimum,
            calibrated=tuple(
                None if panel is None else panel.quantiles for panel in (calibrated, adjusted)
            ),
        )
    return result


def _score_window(sources, config, window_id, ablation=None):
    """Puntuar todos los brazos de una ventana. Cada calibrador se fija antes de evaluar.

    Devuelve las puntuaciones por brazo y semilla y, si se declaran estratos, el registro
    de presencia de la ventana. Con `ablation`, cada brazo con predicciones añade las
    puntuaciones de sus variantes enmascaradas.
    """
    window = sources["windows"][window_id]
    minimum = config["metrics"]["rank_ic_min_assets"]
    calibration = config["calibration"]
    results, reference, calibration_reference = {}, None, None
    row_codes, presence, bits = None, None, None
    for name, arm in config["arms"].items():
        quantile = arm["output"] == QUANTILE_HEAD
        columns = COLUMNS + (QUANTILE_COLUMNS if quantile else ())
        for seed in arm["seeds"]:
            label = f"{name} semilla {seed} en {window_id}"
            files = sources["files"][name, seed, window_id]
            entry = dict(calibrator=None, calibrated=None, calibrated_reason=None)
            excluded = {}
            if quantile:
                table = _read_predictions(files["calibration"], columns)
                table, excluded["calibration"] = _restricted(table, sources, window_id, name)
                fitted = _panel(table, window, "calibration", sources, label, output=QUANTILE_HEAD)
                calibration_reference = calibration_reference or (label, fitted)
                _same_rows(calibration_reference, fitted, f"La calibración de {label}")
                record = cqr.fit_conformal_quantiles(
                    fitted.target,
                    fitted.quantiles,
                    np.asarray(fitted.markets)[fitted.market],
                    levels=LEVELS,
                    nominals=calibration["nominals"],
                    min_rows=calibration["min_rows"],
                )
                # El registro queda congelado con su huella antes de abrir la evaluación.
                entry["calibrator"] = dict(sha256=cqr.calibrator_sha256(record), record=record)
            table = _read_predictions(files["evaluation"], columns)
            table, excluded["evaluation"] = _restricted(table, sources, window_id, name)
            if any(excluded.values()):
                entry["excluded_rows"] = excluded
            panel = _panel(table, window, "evaluation", sources, label, output=arm["output"])
            if reference is None:
                reference = (label, panel)
                targets = table.select(["asset_id", "market", "prediction_at", "target"])
                if STRATA_FIELD in config:
                    bits, presence = _presence_bits(sources, config, window_id, reference)
                    if not presence["incomplete"]:
                        row_codes = modality_strata.codes(bits)
            _same_rows(reference, panel, label)
            entry["raw"] = score_sessions(panel, rank_ic_min_assets=minimum)
            calibrated = None
            if quantile:
                calibrated, adjusted, reason = _calibrated(panel, entry["calibrator"]["record"])
                entry.update(calibrated_reason=reason, order_adjusted_rows=adjusted)
                if calibrated is not None:
                    entry["calibrated"] = score_sessions(calibrated, rank_ic_min_assets=minimum)
            if row_codes is not None:
                # Los cuantiles calibrados son los del calibrador común, sin reajuste.
                entry["strata"] = modality_strata.score_strata(
                    panel,
                    row_codes,
                    rank_ic_min_assets=minimum,
                    calibrated=None if calibrated is None else calibrated.quantiles,
                )
            if ablation is not None:
                record = entry["calibrator"]["record"] if quantile else None
                context = (name, arm["output"], columns, bits, reference, minimum)
                entry["ablation"] = _masked_scores(
                    ablation, sources, window_id, name, seed, panel, calibrated, record, context
                )
            results[name, seed] = entry
    for name, arm in config["arms"].items():
        if arm["output"] != ZERO_CONTROL:
            continue
        zeros = np.zeros(targets.num_rows)
        label = f"{name} en {window_id}"
        panel = _panel(
            targets, window, "evaluation", sources, label, output=POINT, prediction=zeros
        )
        _same_rows(reference, panel, label)
        results[name, None] = dict(
            raw=score_sessions(panel, rank_ic_min_assets=minimum),
            calibrator=None,
            calibrated=None,
            calibrated_reason=None,
        )
        if row_codes is not None:
            results[name, None]["strata"] = modality_strata.score_strata(
                panel, row_codes, rank_ic_min_assets=minimum
            )
    return results, presence


def _views(scores, markets):
    """Sesiones de todos los mercados del ámbito y de cada mercado por separado."""
    views = {"+".join(markets): scores}
    if len(markets) > 1:
        for code, market in enumerate(scores.markets):
            views[market] = scores.select_sessions(scores.session_market == code, label=market)
    return views


def _metric_available(scores, metric):
    quantile = metric in QUANTILE_SERIES or metric.startswith(INTERVAL_SCORE)
    return not quantile or scores.levels is not None


def _seed_series(overall, arm, view, metric):
    return SessionSeries.average([scores[view].series(metric) for scores in overall[arm]])


def _contrasts(config, overall, views):
    comparison = config["comparison"]
    options = dict(
        block_length=comparison["block_length"],
        replicates=comparison["replicates"],
        seed=comparison["seed"],
        confidence=comparison["confidence"],
        market_weighting=config["metrics"]["market_weighting"],
        sensitivity_block_lengths=tuple(comparison["sensitivity_block_lengths"]),
    )
    result = {}
    for view in views:
        result[view] = {}
        for family, contrasts in config["resolved_families"].items():
            arms = sorted({arm for c in contrasts.values() for arm in c})
            rows = {}
            for metric in comparison["metrics"]:
                if not all(_metric_available(overall[arm][0][view], metric) for arm in arms):
                    rows[metric] = dict(reason="Algún brazo de la familia no emite cuantiles")
                    continue
                series = {arm: _seed_series(overall, arm, view, metric) for arm in arms}
                rows[metric] = compare_series(series, contrasts, **options)
            result[view][family] = rows
    return result


def _status(row):
    joint = row["simultaneous_interval"]
    if joint is None:
        return None
    return "undercovers" if joint[1] < 0 else ("overcovers" if joint[0] > 0 else "inconclusive")


def _interval_calibration(config, overall, calibrated, views):
    """Cobertura observada menos nominal con y sin calibración, con intervalo por bloques."""
    comparison = config["comparison"]
    result = {}
    for arm, items in overall.items():
        if items[0][next(iter(views))].levels is None:
            continue
        result[arm] = {}
        for view in views:
            result[arm][view] = {}
            for nominal in config["calibration"]["nominals"]:
                metric = f"{COVERAGE_ERROR}{nominal:g}"
                series = {"raw": _seed_series(overall, arm, view, metric)}
                # Una semilla sin calibración completa deja el brazo sin cobertura calibrada.
                if None not in calibrated[arm]:
                    series["calibrated"] = _seed_series(calibrated, arm, view, metric)
                contrasts = {name: level(name) for name in series}
                if "calibrated" in series:
                    contrasts["calibrated-raw"] = delta("raw", "calibrated")
                compared = compare_series(
                    series,
                    contrasts,
                    block_length=comparison["block_length"],
                    replicates=comparison["replicates"],
                    seed=comparison["seed"],
                    confidence=comparison["confidence"],
                    market_weighting=config["metrics"]["market_weighting"],
                )
                rows = {row["name"]: row for row in compared["contrasts"]}
                result[arm][view][f"{nominal:g}"] = dict(
                    nominal=nominal,
                    raw_coverage_error=rows["raw"]["estimate"],
                    raw_status=_status(rows["raw"]),
                    calibrated_coverage_error=(
                        rows["calibrated"]["estimate"] if "calibrated" in rows else None
                    ),
                    calibrated_status=(
                        _status(rows["calibrated"]) if "calibrated" in rows else None
                    ),
                    calibrated_reason=(
                        None if "calibrated" in rows else "Alguna ventana no tiene calibrador"
                    ),
                    comparison=compared,
                )
    return result


def _period_bins(scores):
    """Filas, probabilidad y subidas por día UTC e intervalo de probabilidad, [días, bins]."""
    periods = int(scores.session_period.max()) + 1
    cells = []
    for values in (scores.sign_bin_rows, scores.sign_bin_probability, scores.sign_bin_up):
        total = np.zeros((periods, SIGN_BINS))
        np.add.at(total, scores.session_period, values)
        cells.append(total)
    return cells


def _ece(items, comparison):
    """ECE medio de las semillas, su curva conjunta y su intervalo percentil por bloques.

    Las semillas comparten días y réplicas, así que cada réplica promedia los ECE de las
    semillas con los mismos días remuestreados, como los contrastes de ``compare_series``.
    El ECE tiene sesgo positivo con pocas filas por intervalo y el intervalo lo describe,
    no lo corrige.
    """
    cells = [_period_bins(scores) for scores in items]
    totals = [[cell.sum(axis=0) for cell in seed] for seed in cells]
    estimates = [expected_calibration_error(*seed) for seed in totals]
    rows, probability, up = (np.sum([seed[i] for seed in totals], axis=0) for i in range(3))
    result = dict(
        estimate=None if None in estimates else float(np.mean(estimates)),
        per_seed=estimates,
        reliability=[
            dict(
                lower=index / SIGN_BINS,
                upper=(index + 1) / SIGN_BINS,
                rows=int(rows[index]),
                mean_probability=float(probability[index] / rows[index]) if rows[index] else None,
                observed_up_frequency=float(up[index] / rows[index]) if rows[index] else None,
            )
            for index in range(SIGN_BINS)
        ],
        interval=None,
        reason=None,
    )
    periods, block = cells[0][0].shape[0], comparison["block_length"]
    if result["estimate"] is None:
        result["reason"] = "Alguna semilla no tiene filas con objetivo no nulo"
        return result
    if block >= periods:
        result["reason"] = "Se necesitan más días que la longitud del bloque"
        return result
    rng = np.random.default_rng(comparison["seed"])
    draws = []
    for offset in range(0, comparison["replicates"], 256):
        size = min(256, comparison["replicates"] - offset)
        counts = circular_block_counts(rng, size, periods, block).astype(np.float64)
        values = [expected_calibration_error(*(counts @ cell for cell in seed)) for seed in cells]
        draws.append(np.mean(values, axis=0))
    draws = np.concatenate(draws)
    if np.isnan(draws).any():
        result["reason"] = "Alguna réplica no contiene filas con objetivo no nulo"
        return result
    tail = (1 - comparison["confidence"]) / 2
    result["interval"] = [float(v) for v in np.quantile(draws, [tail, 1 - tail])]
    return result


def _sign_reliability(config, overall, calibrated, views):
    """Fiabilidad de la probabilidad implícita de subida por brazo y vista, bruta y calibrada."""
    comparison = config["comparison"]
    resampling = dict(
        method="circular_block_bootstrap",
        unit="utc_calendar_day_with_all_sessions_and_assets",
        block_length=comparison["block_length"],
        replicates=comparison["replicates"],
        seed=comparison["seed"],
        confidence=comparison["confidence"],
        interval="percentile_marginal",
    )
    result = {}
    for arm, items in overall.items():
        if items[0][next(iter(views))].levels is None:
            continue
        result[arm] = {}
        for view in views:
            entry = dict(raw=_ece([scores[view] for scores in items], comparison))
            adjusted = calibrated.get(arm, [None])
            entry["calibrated"] = (
                None
                if None in adjusted
                else _ece([scores[view] for scores in adjusted], comparison)
            )
            entry["calibrated_reason"] = (
                "Alguna ventana no tiene calibrador" if entry["calibrated"] is None else None
            )
            result[arm][view] = entry
    return dict(resampling=resampling, bins=SIGN_BINS, arms=result)


def _session_tables(windows, arm, seed):
    tables = []
    for window_id, entry in windows.items():
        for variant in ("raw", "calibrated"):
            if entry[variant] is None:
                continue
            table = entry[variant].to_table()
            for column, value in (
                ("arm", arm),
                ("seed", seed),
                ("window", window_id),
                ("quantiles", variant),
            ):
                table = table.append_column(column, pa.array([value] * len(table)))
            tables.append(table)
    return tables


def _arm_summary(windows, views, calibrated_views, missing, weighting):
    def summary(scores):
        return None if scores is None else scores.summary(market_weighting=weighting)

    return dict(
        windows={
            window_id: dict(
                summary=summary(entry["raw"]),
                calibrated=summary(entry["calibrated"]),
                calibrated_reason=entry["calibrated_reason"],
                calibrator=entry["calibrator"],
                order_adjusted_rows=entry.get("order_adjusted_rows"),
                **({"excluded_rows": entry["excluded_rows"]} if "excluded_rows" in entry else {}),
            )
            for window_id, entry in windows.items()
        },
        overall={
            view: dict(
                summary=summary(scores),
                calibrated=None if calibrated_views is None else summary(calibrated_views[view]),
            )
            for view, scores in views.items()
        },
        missing_calibration_windows=missing,
    )


def _stratum_arm(config, per_window, key, stratum, markets, names):
    """Vistas en bruto y calibradas de un brazo y semilla en un estrato y sus partes."""
    parts = [results[key]["strata"][stratum] for results in per_window.values()]
    raw = [part["raw"] for part in parts if part["raw"] is not None]
    views = modality_strata.subset_views(
        SessionScores.concatenate(raw) if raw else None, markets, names
    )
    calibrated = None
    # Sin calibrador en alguna ventana, el estrato tampoco tiene cobertura calibrada.
    if config["arms"][key[0]]["output"] == QUANTILE_HEAD and all(
        results[key]["calibrated"] is not None for results in per_window.values()
    ):
        adjusted = [part["calibrated"] for part in parts if part["calibrated"] is not None]
        joined = SessionScores.concatenate(adjusted) if adjusted else None
        calibrated = modality_strata.subset_views(joined, markets, names)
    return views, calibrated, parts


def _stratum_summary(scores, calibrated, quantile, cell, weighting):
    """MAE por sesión y cobertura y anchura calibradas de una celda estimable."""
    result = dict(
        session_mae=None, reason=cell["reason"], calibrated_intervals=None, calibrated_reason=None
    )
    if not cell["estimable"]:
        result["calibrated_reason"] = cell["reason"]
        return result
    result["session_mae"] = scores.summary(market_weighting=weighting)["point"]["mae"]
    if not quantile:
        result["calibrated_reason"] = "El brazo no emite cuantiles"
    elif calibrated is None:
        result["calibrated_reason"] = "Alguna ventana no tiene calibrador"
    else:
        intervals = calibrated.summary(market_weighting=weighting)["quantiles"]["intervals"]
        result["calibrated_intervals"] = [
            {key: row[key] for key in ("nominal", "coverage", "coverage_gap", "width")}
            for row in intervals
        ]
    return result


def _window_mae(parts, cells, weighting):
    """MAE por sesión de cada ventana y mercado estimables, y None en los demás."""
    result = {}
    for (window, row), part in zip(cells.items(), parts, strict=True):
        by_market = {}
        if any(cell["estimable"] for cell in row.values()):
            by_market = part["raw"].summary(market_weighting=weighting)["by_market"]
        result[window] = {
            market: by_market[market]["mae"] if cell["estimable"] else None
            for market, cell in row.items()
        }
    return result


def _strata_report(config, scored, overall, markets):
    """Sección secundaria por estrato de presencia, con los umbrales y la confianza declarados.

    Las celdas por debajo del umbral aparecen con su motivo y sin métricas. La confianza de
    contrastes y coberturas corrige por Bonferroni el número de celdas de estrato y ámbito.
    Si alguna ventana tiene filas sin precios, gráficos y macro, la sección entera queda no
    estimable con sus recuentos. La métrica principal no depende de este resultado.
    """
    declared = config[STRATA_FIELD]
    presence = {window: record for window, (_, record) in scored.items()}
    incomplete = {window: record["incomplete"] for window, record in presence.items()}
    if any(incomplete.values()):
        return dict(
            declaration=declared,
            status="not_estimable",
            reason=(
                f"{sum(incomplete.values())} filas de evaluación no tienen precios, gráficos y "
                "macro, así que los estratos declarados no describen la población"
            ),
            presence=presence,
        )
    per_window = {window: results for window, (results, _) in scored.items()}
    thresholds = {key: declared[key] for key in ("min_rows", "min_sessions")}
    weighting = config["metrics"]["market_weighting"]
    keys = list(per_window[next(iter(per_window))])
    reference = overall[keys[0][0]][0]
    names = list(reference)
    cells = len(modality_strata.STRATA) * len(names)
    confidence = modality_strata.adjusted_confidence(config["comparison"]["confidence"], cells)
    local = dict(
        config,
        comparison=dict(config["comparison"], metrics=declared["metrics"], confidence=confidence),
    )
    population, arms, contrasts, calibration = {}, {}, {}, {}
    for stratum, pattern in modality_strata.STRATA.items():
        by_key = {
            key: _stratum_arm(config, per_window, key, stratum, markets, names) for key in keys
        }
        # Todos los brazos evalúan las mismas filas, así que la población es la del primero.
        first_views, _, first_parts = by_key[keys[0]]
        whole = {}
        for view in names:
            cell = modality_strata.cell(first_views[view], thresholds)
            whole[view] = dict(cell, row_share=cell["rows"] / int(reference[view].samples.sum()))
        windows = {
            window: {
                market: modality_strata.cell(part["raw"], thresholds, market) for market in markets
            }
            for window, part in zip(per_window, first_parts, strict=True)
        }
        population[stratum] = dict(pattern=pattern, overall=whole, windows=windows)
        views, calibrated = {}, {}
        for (arm, seed), (raw, adjusted, parts) in by_key.items():
            quantile = config["arms"][arm]["output"] == QUANTILE_HEAD
            views.setdefault(arm, []).append(raw)
            if quantile:
                calibrated.setdefault(arm, []).append(adjusted)
            label = "deterministic" if seed is None else str(seed)
            arms.setdefault(arm, {}).setdefault(label, {})[stratum] = dict(
                overall={
                    view: _stratum_summary(
                        raw[view],
                        None if adjusted is None else adjusted[view],
                        quantile,
                        whole[view],
                        weighting,
                    )
                    for view in names
                },
                windows=_window_mae(parts, windows, weighting),
            )
        estimable = [view for view in names if whole[view]["estimable"]]
        compared = _contrasts(local, views, estimable) if estimable else {}
        intervals = _interval_calibration(local, views, calibrated, estimable) if estimable else {}
        missing = {view: dict(reason=whole[view]["reason"]) for view in names}
        contrasts[stratum] = {view: compared.get(view, missing[view]) for view in names}
        calibration[stratum] = {
            arm: {view: intervals.get(arm, {}).get(view, missing[view]) for view in names}
            for arm in calibrated
        }
    return dict(
        declaration=declared,
        status="computed",
        reason=None,
        presence=presence,
        recalibrated=False,
        multiplicity=dict(
            method=modality_strata.MULTIPLICITY,
            cells=cells,
            family_confidence=config["comparison"]["confidence"],
            confidence=confidence,
        ),
        population=population,
        arms=arms,
        contrasts=contrasts,
        interval_calibration=calibration,
    )


def evaluate_walk_forward(
    config_path, sources_path, scope, *, ablation_sources=None, aggregates=None
):
    """Calcular el informe y la tabla por sesión de un ámbito sin escribir nada.

    `config_path` es la ruta de la configuración o una configuración ya validada.
    `ablation_sources` es el manifiesto de la etapa de ablación de modalidades. Solo se
    admite si la configuración declara la ablación. `aggregates` es la carpeta de los
    agregados por ventana (`window_aggregates`) que guardó la retención v2. Con ella no se
    lee ninguna predicción por fila, y cada ventana exige agregados de estas mismas fuentes.
    """
    started = time.perf_counter()
    config = resolve_config(config_path)
    sources = load_sources(sources_path, config, scope)
    # Desde aquí, los brazos y las familias son los del ámbito evaluado.
    config = scope_config(config, scope)
    weighting, markets = config["metrics"]["market_weighting"], sources["markets"]
    ablation = None
    if ablation_sources is not None:
        _require(ABLATION_FIELD in config, "La configuración no declara la ablación de modalidades")
        ablation = _ablation_sources(ablation_sources, config, sources)
    if aggregates is None:
        scored = {
            window: _score_window(sources, config, window, ablation)
            for window in sources["windows"]
        }
    else:
        from . import window_aggregates

        scored = {
            window: window_aggregates.read(aggregates, config, sources, window, ablation)
            for window in sources["windows"]
        }
    per_window = {window: results for window, (results, _) in scored.items()}
    overall, calibrated, arms, tables = {}, {}, {}, []
    for arm, seed in per_window[next(iter(per_window))]:
        windows = {window: results[arm, seed] for window, results in per_window.items()}
        quantile = config["arms"][arm]["output"] == QUANTILE_HEAD
        views = _views(SessionScores.concatenate([e["raw"] for e in windows.values()]), markets)
        missing = [window for window, e in windows.items() if quantile and e["calibrated"] is None]
        calibrated_views = None
        if quantile and not missing:
            joined = SessionScores.concatenate([e["calibrated"] for e in windows.values()])
            calibrated_views = _views(joined, markets)
        overall.setdefault(arm, []).append(views)
        if quantile:
            calibrated.setdefault(arm, []).append(calibrated_views)
        record = arms.setdefault(arm, dict(output=config["arms"][arm]["output"], seeds={}))
        record["seeds"]["deterministic" if seed is None else str(seed)] = _arm_summary(
            windows, views, calibrated_views, missing, weighting
        )
        tables.extend(_session_tables(windows, arm, seed))
    names = list(overall[next(iter(overall))][0])
    strata = _strata_report(config, scored, overall, markets) if STRATA_FIELD in config else None
    report = dict(
        schema_version=1,
        kind=REPORT_KIND,
        status="completed",
        created_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        scope=scope,
        markets=markets,
        configuration=dict(name=config["name"], sha256=config["sha256"]),
        sources_sha256=sources["sha256"],
        **(policy_identity(config["input_policy"]) or dict(input_policy=config["input_policy"])),
        edition=sources["edition"],
        protocol_sha256=sources["protocol_sha256"],
        windows={
            window_id: dict(
                calibration=window["calibration"],
                evaluation=window["evaluation"],
                view_sha256=sources["views"][window_id],
            )
            for window_id, window in sources["windows"].items()
        },
        metrics=config["metrics"],
        calibration=config["calibration"],
        abstention=ABSTENTION,
        aggregation=dict(
            session="mean_over_assets_of_one_market_and_decision_instant",
            across_windows="pooled_sessions_of_all_declared_windows",
            seeds="summaries_per_seed_and_session_mean_series_for_contrasts",
            market_weighting=weighting,
        ),
        arms=arms,
        contrasts=_contrasts(config, overall, names),
        interval_calibration=_interval_calibration(config, overall, calibrated, names),
        sign_reliability=_sign_reliability(config, overall, calibrated, names),
        versions={name: version(name) for name in ("numpy", "pyarrow")},
        analysis_source_sha256={
            name: sha256(Path(__file__).parents[1] / name)
            for name in (
                "evaluation/walk_forward_comparison.py",
                "evaluation/forecast_panel.py",
                "evaluation/forecast_scores.py",
                "evaluation/paired_comparisons.py",
                "calibration/conformal_quantiles.py",
                "training/temporal_contract.py",
            )
        },
        resources=dict(
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            gpu_used=False,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    if JOINT_FIELD in config:
        report[JOINT_FIELD] = dict(
            declaration=config[JOINT_FIELD],
            eligible_windows=sources["eligible"],
            eligibility_sha256=sources.get("eligibility_sha256"),
            borrowed=sources["borrowed"],
            joint_windows=sources["joint_windows"],
            joint_views=sources["joint_views"],
        )
    if strata is not None:
        report[STRATA_FIELD] = strata
        for name in ("evaluation/modality_strata.py", "training/corpus_inputs.py"):
            report["analysis_source_sha256"][name] = sha256(Path(__file__).parents[1] / name)
    if ABLATION_FIELD in config:
        report[ABLATION_FIELD] = (
            modality_ablation.pending(config)
            if ablation is None
            else modality_ablation.report(
                config,
                {
                    window: {
                        key: entry["ablation"]
                        for key, entry in results.items()
                        if key[1] is not None
                    }
                    for window, results in per_window.items()
                },
                markets,
                names,
                sources_sha256=ablation["sha256"],
            )
        )
        for name in ("evaluation/modality_ablation.py", "data/modality_ablation.py"):
            report["analysis_source_sha256"][name] = sha256(Path(__file__).parents[1] / name)
    return report, pa.concat_tables(tables, promote_options="default")


def write_walk_forward(
    config_path, sources_path, scope, output, *, ablation_sources=None, aggregates=None
):
    """Publicar el informe y las sesiones en un directorio nuevo fuera de las fuentes."""
    output = Path(output)
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    folders = [Path(config_path).parent, Path(sources_path).parent]
    if ablation_sources is not None:
        folders.append(Path(ablation_sources).parent)
    for source in folders:
        outside_source(source, output)
        outside_source(output, source)
    report, sessions = evaluate_walk_forward(
        config_path, sources_path, scope, ablation_sources=ablation_sources, aggregates=aggregates
    )
    json.dumps(report, allow_nan=False)
    output.mkdir(parents=True)
    pq.write_table(sessions, output / "sessions.parquet", compression="zstd")
    report["artifacts"] = {"sessions.parquet": sha256(output / "sessions.parquet")}
    atomic_json(output / "comparison.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(SCOPES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ablation-sources", type=Path)
    parser.add_argument("--aggregates", type=Path, help="Agregados por ventana de la retención v2")
    args = parser.parse_args(argv)
    report = write_walk_forward(
        args.config,
        args.sources,
        args.scope,
        args.output,
        ablation_sources=args.ablation_sources,
        aggregates=args.aggregates,
    )
    windows = len(report["windows"])
    print(f"Comparados {len(report['arms'])} brazos en {windows} ventanas. Reserva final cerrada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
