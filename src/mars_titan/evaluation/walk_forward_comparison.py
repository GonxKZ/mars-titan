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
from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import masked_inputs, policy_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import modality_strata
from mars_titan.evaluation.forecast_panel import WEIGHTINGS, ForecastPanel, SessionSeries
from mars_titan.evaluation.forecast_scores import COVERAGE_ERROR, SessionScores, score_sessions
from mars_titan.evaluation.paired_comparisons import compare_series, delta, interaction, level
from mars_titan.evaluation.splits import build_folds
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
SERIES_METRICS = ("mae", "mse", "direction_accuracy", "rank_ic", "pinball")
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


def load_config(path):
    """Validar la configuración declarada antes de abrir ninguna predicción."""
    path = Path(path)
    config, digest = read_manifest(path, 1024**2)
    version = config.get("schema_version") if isinstance(config, dict) else None
    _require(
        isinstance(config, dict)
        and type(version) is int
        and version in (1, 2)
        and set(config) == _CONFIG_FIELDS | ({STRATA_FIELD} if version == 2 else set())
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
    if version == 2:
        _require(
            masked_inputs(config["input_policy"]),
            "Los estratos de presencia necesitan la política de entradas con máscaras",
        )
        modality_strata.declaration(config[STRATA_FIELD], SERIES_METRICS)
    resolved = {
        scope: _protocols(path.parent, scope, declared) for scope, declared in scopes.items()
    }
    return dict(config, sha256=digest, resolved_scopes=resolved, resolved_families=families)


def _file(folder, record, label):
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
    views, edition = {}, None
    for window_id, window in windows.items():
        entry = sources["windows"][window_id]
        _require(isinstance(entry, dict) and set(entry) == {"view"}, "La ventana necesita su vista")
        views[window_id], current = _view(path.parent, entry["view"], window, scope, policy)
        _require(
            edition is None or current == edition, "Las ventanas no comparten la misma edición"
        )
        edition = current
    declared = {name: arm for name, arm in config["arms"].items() if arm["output"] != ZERO_CONTROL}
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
                _require(entry["view_sha256"] == views[window_id], f"{where} usa otra vista")
                files[name, int(seed), window_id] = {
                    part: _file(path.parent, entry[part], f"{where} ({part})")
                    for part in ("calibration", "evaluation")
                    if part in entry
                }
    return dict(sha256=digest, scope=scope_name, views=views, edition=edition, files=files, **scope)


def _read_predictions(file, columns):
    """Leer solo las columnas necesarias después de comprobar huella y tipos."""
    path = file["path"]
    _require(sha256(path) == file["sha256"], f"La huella de {path.name} no coincide")
    schema = pq.read_schema(path)
    _require(set(columns) <= set(schema.names), f"Faltan columnas en {path.name}")
    _require(
        schema.field("prediction_at").type == pa.timestamp("us", tz="UTC"),
        "Los instantes deben ser timestamp UTC en microsegundos",
    )
    table = pq.read_table(path, columns=list(columns), use_threads=False)
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


def _panel(table, window, partition, scope, label, *, output, prediction=None):
    market, times = _check_segment(table, window, partition, scope["markets"], label)
    # La identidad de fila es (mercado, activo, instante), común a todos los runners.
    row_id = pc.binary_join_element_wise(
        table["market"].cast(pa.large_string()),
        table["asset_id"].cast(pa.large_string()),
        table["prediction_at"].cast(pa.int64()).cast(pa.large_string()),
        pa.scalar("/", pa.large_string()),
    )
    quantile = output == QUANTILE_HEAD
    return ForecastPanel.from_columns(
        row_id,
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


def _score_window(sources, config, window_id):
    """Puntuar todos los brazos de una ventana. Cada calibrador se fija antes de evaluar."""
    window = sources["windows"][window_id]
    minimum = config["metrics"]["rank_ic_min_assets"]
    calibration = config["calibration"]
    results, reference, calibration_reference = {}, None, None
    for name, arm in config["arms"].items():
        quantile = arm["output"] == QUANTILE_HEAD
        columns = COLUMNS + (QUANTILE_COLUMNS if quantile else ())
        for seed in arm["seeds"]:
            label = f"{name} semilla {seed} en {window_id}"
            files = sources["files"][name, seed, window_id]
            entry = dict(calibrator=None, calibrated=None, calibrated_reason=None)
            if quantile:
                table = _read_predictions(files["calibration"], columns)
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
            panel = _panel(table, window, "evaluation", sources, label, output=arm["output"])
            if reference is None:
                reference = (label, panel)
                targets = table.select(["asset_id", "market", "prediction_at", "target"])
            _same_rows(reference, panel, label)
            entry["raw"] = score_sessions(panel, rank_ic_min_assets=minimum)
            if quantile:
                calibrated, adjusted, reason = _calibrated(panel, entry["calibrator"]["record"])
                entry.update(calibrated_reason=reason, order_adjusted_rows=adjusted)
                if calibrated is not None:
                    entry["calibrated"] = score_sessions(calibrated, rank_ic_min_assets=minimum)
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
    return results


def _views(scores, markets):
    """Sesiones de todos los mercados del ámbito y de cada mercado por separado."""
    views = {"+".join(markets): scores}
    if len(markets) > 1:
        for code, market in enumerate(scores.markets):
            views[market] = scores.select_sessions(scores.session_market == code, label=market)
    return views


def _metric_available(scores, metric):
    return metric != "pinball" or scores.levels is not None


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


def evaluate_walk_forward(config_path, sources_path, scope):
    """Calcular el informe y la tabla por sesión de un ámbito sin escribir nada."""
    started = time.perf_counter()
    config = load_config(config_path)
    sources = load_sources(sources_path, config, scope)
    weighting, markets = config["metrics"]["market_weighting"], sources["markets"]
    per_window = {window: _score_window(sources, config, window) for window in sources["windows"]}
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
    return report, pa.concat_tables(tables, promote_options="default")


def write_walk_forward(config_path, sources_path, scope, output):
    """Publicar el informe y las sesiones en un directorio nuevo fuera de las fuentes."""
    output = Path(output)
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    for source in (Path(config_path).parent, Path(sources_path).parent):
        outside_source(source, output)
        outside_source(output, source)
    report, sessions = evaluate_walk_forward(config_path, sources_path, scope)
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
    args = parser.parse_args(argv)
    report = write_walk_forward(args.config, args.sources, args.scope, args.output)
    windows = len(report["windows"])
    print(f"Comparados {len(report['arms'])} brazos en {windows} ventanas. Reserva final cerrada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
