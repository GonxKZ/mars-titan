"""Filas y métricas de una comparación walk-forward comprobadas por otro camino al publicar.

La publicación de la campaña no se escribe si esta comprobación falla. Según dónde estén las
predicciones por fila hay dos caminos:

- Con las filas presentes o compactadas sin pérdida, `published` aplica a la comparación
  publicada `row_identity.check_loaded` y `score_recheck`, que leen cada archivo de
  predicciones con su huella.
- Con la retención v2, las filas de una ventana se liberan después de guardar sus agregados.
  Por eso `write_window` comprueba la ventana justo después de guardarlos y antes de liberar
  nada: identidad de filas entre modelos, filas quitadas por el diseño y sesiones recalculadas
  desde las filas frente a las sesiones en bruto de los agregados. El registro, junto a los
  agregados, guarda su huella y su identidad. Al publicar, `published` exige el registro
  superado de cada ventana con esa misma huella e identidad, que las sesiones publicadas
  sean exactamente las de los agregados y que las métricas del informe se vuelvan a obtener
  desde ellas.

En los dos caminos se cotejan con el recuento propio las filas que la comparación dice haber
quitado de cada ventana, y la abstención publicada con su cobertura (`abstention_marks`).
Nada de este módulo ajusta modelos ni abre la reserva de 2024.
"""

import json
from pathlib import Path

import pandas as pd

from mars_titan.data.storage import atomic_json, sha256
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.evaluation import window_aggregates

from . import abstention_marks, metric_registry, row_identity, score_recheck

KIND = "comparison_integrity"
WINDOW_KIND = "comparison_window_integrity"
SCHEMA_VERSION = 1


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def record_path(aggregates_path):
    """Registro de integridad de unos agregados de ventana, en su misma carpeta."""
    path = Path(aggregates_path)
    return path.with_name(f"{path.stem}.integrity.json")


def _seed_key(seed):
    return "deterministic" if seed is None else str(seed)


def _aggregate_raw(results, window_id):
    """Sesiones en bruto de los agregados de una ventana, con modelo, semilla y ventana."""
    frames = []
    for (arm, seed), entry in results.items():
        frame = entry["raw"].to_table().to_pandas()
        frame["arm"], frame["seed"], frame["window"] = arm, seed, window_id
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _declared_exclusions(results, window_id):
    """Filas quitadas por la comparación en cada archivo, como las publica el informe."""
    return {
        f"{arm}/{seed}/{window_id}/{partition}": counts
        for (arm, seed), entry in results.items()
        if seed is not None
        for partition, counts in entry.get("excluded_rows", {}).items()
    }


def _report_exclusions(report, sources):
    """Lo mismo desde el resumen por ventana de cada modelo y semilla del informe publicado."""
    found = {}
    for arm, seed, window_id in sources["files"]:
        windows = report["arms"][arm]["seeds"][_seed_key(seed)]["windows"]
        for partition, counts in windows[window_id].get("excluded_rows", {}).items():
            found[f"{arm}/{seed}/{window_id}/{partition}"] = counts
    return found


def _counted(rows):
    """Filas quitadas por el diseño que contó `row_identity` en cada archivo que las tiene."""
    return {
        f"{entry['arm']}/{entry['seed']}/{entry['window']}/{entry['partition']}": entry[
            "excluded_rows"
        ]
        for entry in rows["files"]
        if entry.get("excluded_rows")
    }


def _exclusions(counted, declared):
    """Cotejar las filas quitadas contadas con las que declara la comparación.

    Un archivo sin filas quitadas por el diseño no aparece en ninguno de los dos lados.
    """
    keys = counted.keys() | declared.keys()
    differing = sorted(key for key in keys if counted.get(key) != declared.get(key))
    return dict(files=len(counted), differing=differing, passed=not differing)


def _scores(checks, failures):
    return dict(files=checks, failures=failures, passed=failures == 0)


def write_window(folder, config, sources, window_id, ablation=None, *, liquidity=None):
    """Comprobar la ventana recién agregada antes de liberar sus filas y guardar el registro.

    `folder`, `config`, `sources`, `ablation` y `liquidity` son los de
    `window_aggregates.write`, con las fuentes limitadas a la ventana. El registro se guarda
    también cuando falla, para revisarlo, y en ese caso la llamada termina con un error y la
    ventana no llega a liberarse.
    """
    _require(set(sources["windows"]) == {window_id}, "Las fuentes deben limitarse a la ventana")
    path = window_aggregates.path_for(folder, sources["scope"], window_id)
    results = window_aggregates.read(
        folder, config, sources, window_id, ablation, liquidity=liquidity
    )[0]
    rows = row_identity.check_loaded(config, sources)
    raw = _aggregate_raw(results, window_id)
    checks, failures = score_recheck.check_files(config, sources, raw)
    excluded = _exclusions(_counted(rows), _declared_exclusions(results, window_id))
    passed = rows["passed"] and failures == 0 and excluded["passed"]
    record = dict(
        schema_version=SCHEMA_VERSION,
        kind=WINDOW_KIND,
        scope=sources["scope"],
        window=window_id,
        aggregates=dict(path=path.name, sha256=sha256(path)),
        identity=window_aggregates.identity(
            config, sources, window_id, ablation, liquidity=liquidity
        ),
        tolerance=dict(rtol=score_recheck.RTOL, atol=score_recheck.ATOL),
        rows=rows,
        scores=_scores(checks, failures),
        excluded_rows=excluded,
        passed=passed,
        training_executed=False,
        final_test_opened=False,
    )
    destination = record_path(path)
    atomic_json(destination, record)
    _require(passed, f"La integridad de {sources['scope']} {window_id} falla: revisa {destination}")
    return dict(path=destination, sha256=sha256(destination), passed=passed)


def write_checked(folder, config, sources, window_id, ablation=None, *, liquidity=None):
    """Guardar los agregados de una ventana y su registro de integridad, en ese orden.

    Devuelve los dos registros. Si la comprobación falla, el error llega antes de que nadie
    libere las filas de la ventana.
    """
    stored = window_aggregates.write(
        folder, config, sources, window_id, ablation, liquidity=liquidity
    )
    return stored, write_window(folder, config, sources, window_id, ablation, liquidity=liquidity)


def _window_records(folder, config, sources, ablation, liquidity):
    """Registros de cada ventana, exigiendo que sigan describiendo los agregados actuales.

    Devuelve la huella de cada registro, las filas quitadas que contó y los fallos.
    """
    found, counted, failures = {}, {}, []
    for window_id in sources["windows"]:
        path = window_aggregates.path_for(folder, sources["scope"], window_id)
        destination = record_path(path)
        if not destination.is_file():
            failures.append(f"{window_id}: falta el registro de integridad")
            continue
        record = json.loads(destination.read_text(encoding="utf-8"))
        expected = window_aggregates.identity(
            config, sources, window_id, ablation, liquidity=liquidity
        )
        checks = (
            (record.get("kind") == WINDOW_KIND, "no es un registro de integridad"),
            (record.get("passed") is True, "registra fallos"),
            (
                path.is_file() and record.get("aggregates", {}).get("sha256") == sha256(path),
                "describe otros agregados",
            ),
            (record.get("identity") == expected, "procede de otras fuentes o código"),
        )
        problems = [message for condition, message in checks if not condition]
        failures += [f"{window_id}: el registro {message}" for message in problems]
        found[window_id] = dict(sha256=sha256(destination), passed=not problems)
        if not problems:
            counted.update(_counted(record["rows"]))
    return found, counted, failures


def _same_as_aggregates(folder, config, sources, ablation, liquidity, raw):
    """Sesiones publicadas idénticas, sin tolerancia, a las de los agregados de cada ventana.

    Cada modelo se coteja con las columnas de su propia tabla, de modo que las de cuantiles
    solo cuentan en los modelos que los emiten.
    """
    checks, failures = [], 0
    for window_id in sources["windows"]:
        results = window_aggregates.read(
            folder, config, sources, window_id, ablation, liquidity=liquidity
        )[0]
        for (arm, seed), entry in results.items():
            stored = entry["raw"].to_table().to_pandas()
            # El lado propio del cotejo usa microsegundos enteros, como el recálculo.
            stored["prediction_at"] = score_recheck._microseconds(stored["prediction_at"])
            theirs = (raw["arm"] == arm) & (raw["window"] == window_id)
            if seed is not None:
                theirs &= raw["seed"] == seed
            outcome = score_recheck.compare_sessions(
                stored, raw.loc[theirs, list(stored.columns)], rtol=0.0, atol=0.0
            )
            outcome.update(arm=arm, seed=seed, window=window_id)
            failures += not outcome["passed"]
            checks.append(outcome)
    return checks, failures


def published(
    config, sources_path, scope, folder, *, aggregates=None, ablation_sources=None, edition=None
):
    """Comprobar filas, filas quitadas y métricas de la comparación publicada en `folder`.

    `config` es la ruta o la configuración validada. `aggregates` es la carpeta de agregados
    de la retención v2 de la que se calculó la comparación y `ablation_sources`, el manifiesto
    de la ablación con el que se guardaron, si lo hubo. `edition` es la edición que asignó
    los estratos de liquidez guardados con esos agregados, si la comparación los declara.
    """
    folder = Path(folder)
    report = json.loads((folder / "comparison.json").read_text(encoding="utf-8"))
    config = walk.resolve_config(config)
    _require(report["configuration"]["sha256"] == config["sha256"], "Otra configuración")
    sources = walk.load_sources(sources_path, config, scope)
    _require(report["sources_sha256"] == sources["sha256"], "Otro manifiesto de fuentes")
    scoped = walk.scope_config(config, scope)
    raw = score_recheck.published_raw(folder, report)
    result = dict(
        schema_version=SCHEMA_VERSION,
        kind=KIND,
        scope=scope,
        comparison_sha256=sha256(folder / "comparison.json"),
        configuration_sha256=config["sha256"],
        sources_sha256=sources["sha256"],
        from_aggregates=aggregates is not None,
    )
    if aggregates is None:
        rows = row_identity.check_loaded(config, sources)
        checks, failures = score_recheck.check_files(scoped, sources, raw)
        result.update(rows=rows, scores=_scores(checks, failures))
        passed, counted = rows["passed"] and failures == 0, _counted(rows)
    else:
        folder = Path(aggregates)
        ablation = None
        if ablation_sources is not None:
            ablation = walk._ablation_sources(ablation_sources, scoped, sources)
        liquidity = walk.liquidity_source(scoped, edition)
        records, counted, missing = _window_records(folder, scoped, sources, ablation, liquidity)
        checks, failures = _same_as_aggregates(folder, scoped, sources, ablation, liquidity, raw)
        result.update(
            windows=dict(records=records, failures=missing, passed=not missing),
            scores=_scores(checks, failures),
        )
        passed = not missing and failures == 0
    excluded = _exclusions(counted, _report_exclusions(report, sources))
    registry = metric_registry.load_registry()
    aggregates_check, wrong = score_recheck.check_summaries(scoped, sources, raw, report, registry)
    abstention = abstention_marks.check_report(report, raw)
    result.update(
        excluded_rows=excluded,
        summaries=dict(
            registry=dict(path=registry["path"], sha256=registry["sha256"]),
            views=aggregates_check,
            failures=wrong,
            passed=wrong == 0,
        ),
        abstention=abstention,
        passed=passed and excluded["passed"] and wrong == 0 and abstention["passed"],
        training_executed=False,
        final_test_opened=False,
    )
    return result
