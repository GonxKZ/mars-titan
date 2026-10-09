"""Contrastes declarados sobre las tablas por sesión que ya publican otras comparaciones.

La comparación walk-forward publica por brazo, semilla, ventana y variante de cuantiles los
estadísticos de cada sesión (``sessions.parquet``). La cartera larga y corta publica el
rendimiento neto por sesión y coste. Otras etapas (políticas y diagnósticos) pueden publicar
una tabla larga con una métrica por fila. Este módulo reconstruye a partir de esas tablas
las mismas series por sesión que usa ``compare_series``, sin volver a puntuar predicciones,
y estima familias de contrastes lineales con el remuestreo por bloques de la comparación.

Cada brazo se lee una vez. Si aparece en dos informes, sus sesiones y valores deben
coincidir. Las horas GPU por brazo, medidas o proyectadas, permiten expresar cada efecto
como mejora por hora. Nada de este módulo ajusta modelos ni lee la edición.
"""

import hashlib
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import sha256

from . import long_short
from . import walk_forward_comparison as walk
from .forecast_panel import DAY_MICROSECONDS, SessionSeries
from .forecast_scores import INTERVAL_SCORE, SIGN_BINS, expected_calibration_error
from .long_short_comparison import REPORT_KIND as PORTFOLIO_SOURCE
from .paired_comparisons import circular_block_counts, compare_series, family_intervals

SOURCES_KIND = "comparison_matrix_sources"
HOURS_KIND = "comparison_arm_hours"
WALK_SOURCE = walk.REPORT_KIND
TABLE_SOURCE = "session_metric_table"
SIGN_ECE = "sign_ece"
WALK_METRICS = (*walk.SERIES_METRICS, SIGN_ECE)
# Métricas que solo existen si el brazo emite cuantiles y que la calibración cambia.
QUANTILE_METRICS = ("pinball", "sign_brier", SIGN_ECE)
LOWER_IS_BETTER = ("mae", "mse", "pinball", "sign_brier", SIGN_ECE)
ORIENTATIONS = ("loss", "gain")
TABLE_COLUMNS = ("arm", "seed", "market", "prediction_at", "metric", "value")
# Brazos por grupo del bootstrap de la cartera: acota la memoria sin cambiar las réplicas,
# porque cada brazo se remuestrea con los mismos índices sea cual sea su grupo.
PORTFOLIO_GROUP = 16
_ECE_CHUNK = 256
MAX_REPORT_BYTES = 512 * 1024**2


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def is_text(value):
    return isinstance(value, str) and bool(value.strip())


def is_arm_name(value):
    return isinstance(value, str) and 0 < len(value) <= 96 and value.replace("_", "").isalnum()


def quantile_metric(metric):
    return metric in QUANTILE_METRICS or metric.startswith(INTERVAL_SCORE)


def load_hours(path, scope):
    """Horas GPU por brazo (medidas o proyectadas) y acumuladas con sus padres."""
    document, digest = read_manifest(Path(path), 16 * 1024**2)
    arms = document.get("arms") if isinstance(document, dict) else None
    _require(
        isinstance(document, dict)
        and set(document) == {"schema_version", "kind", "basis", "source", "scope", "arms"}
        and document["schema_version"] == 1
        and document["kind"] == HOURS_KIND
        and document["basis"] in ("measured", "projected")
        and is_text(document["source"])
        and document["scope"] == scope
        and isinstance(arms, dict)
        and arms,
        "El documento de horas no cumple su contrato o es de otro ámbito",
    )
    for arm, entry in arms.items():
        _require(
            is_arm_name(arm)
            and isinstance(entry, dict)
            and set(entry) == {"gpu_hours", "parents"}
            and isinstance(entry["gpu_hours"], (int, float))
            and not isinstance(entry["gpu_hours"], bool)
            and math.isfinite(entry["gpu_hours"])
            and entry["gpu_hours"] >= 0
            and isinstance(entry["parents"], list)
            and all(parent in arms and parent != arm for parent in entry["parents"]),
            f"Las horas de {arm} necesitan un valor no negativo y padres declarados",
        )
    cumulative = {}
    for arm in arms:
        ancestors, pending = set(), list(arms[arm]["parents"])
        while pending:
            parent = pending.pop()
            if parent not in ancestors:
                ancestors.add(parent)
                pending.extend(arms[parent]["parents"])
        _require(arm not in ancestors, f"Los padres de {arm} forman un ciclo")
        cumulative[arm] = arms[arm]["gpu_hours"] + math.fsum(
            arms[parent]["gpu_hours"] for parent in sorted(ancestors)
        )
    return dict(
        basis=document["basis"],
        source=document["source"],
        sha256=digest,
        own={arm: float(entry["gpu_hours"]) for arm, entry in arms.items()},
        cumulative=cumulative,
    )


def _orient(lower_is_better):
    return None if lower_is_better is None else (-1.0 if lower_is_better else 1.0)


def cost_adjusted(row, sign, hours):
    """Mejora por hora GPU de un efecto, con el intervalo simultáneo dividido por las horas.

    ``sign`` vale −1 si menor es mejor y +1 si mayor es mejor. Las horas son las acumuladas
    con los padres, combinadas con los mismos coeficientes del contraste, y se tratan como
    una constante medida. Un nivel (coeficientes que no suman cero) no tiene coste propio.
    """
    coefficients = row["coefficients"]
    result = dict(
        basis=hours["basis"],
        gpu_hours_difference=None,
        improvement=None,
        improvement_per_gpu_hour=None,
        improvement_per_gpu_hour_interval=None,
        reason=None,
    )
    if sign is None:
        result["reason"] = "La métrica no tiene un sentido de mejora declarado"
        return result
    if not math.isclose(math.fsum(coefficients.values()), 0.0, abs_tol=1e-12):
        result["reason"] = "Es un nivel, no un efecto entre brazos"
        return result
    unknown = sorted(arm for arm in coefficients if arm not in hours["cumulative"])
    if unknown:
        result["reason"] = f"Sin horas para {', '.join(unknown)}"
        return result
    difference = math.fsum(c * hours["cumulative"][arm] for arm, c in coefficients.items())
    result["gpu_hours_difference"] = difference
    if row.get("estimate") is None:
        result["reason"] = "El contraste no tiene estimación"
        return result
    result["improvement"] = sign * row["estimate"]
    if difference <= 0:
        result["reason"] = "La variante no consume más horas GPU que su referencia"
        return result
    result["improvement_per_gpu_hour"] = result["improvement"] / difference
    joint = row.get("simultaneous_interval")
    if joint is not None:
        bounds = sorted(sign * value / difference for value in joint)
        result["improvement_per_gpu_hour_interval"] = bounds
    return result


# Fuentes


def load_sources(path, views, config, scope):
    """Validar el manifiesto de fuentes y comprobar que todos los informes son coherentes.

    ``views`` son las vistas declaradas en la matriz y ``config`` la comparación validada.
    """
    path = Path(path)
    manifest, digest = read_manifest(path, 4 * 1024**2)
    _require(
        isinstance(manifest, dict)
        and set(manifest) == {"schema_version", "kind", "scope", "reports", "hours"}
        and manifest["schema_version"] == 1
        and manifest["kind"] == SOURCES_KIND
        and manifest["scope"] == scope
        and scope in config["resolved_scopes"]
        and isinstance(manifest["reports"], list)
        and manifest["reports"],
        "El manifiesto de fuentes de la matriz no cumple su contrato o es de otro ámbito",
    )
    folder = path.parent
    reports = []
    for index, entry in enumerate(manifest["reports"]):
        label = f"La fuente {index}"
        _require(isinstance(entry, dict), f"{label} debe ser un objeto")
        kind = entry.get("kind")
        record = {key: value for key, value in entry.items() if key not in ("kind", "view")}
        file = walk._file(folder, record, label)
        if kind == TABLE_SOURCE:
            view = entry.get("view")
            declared = views.get(view)
            _require(
                set(entry) == {"kind", "path", "sha256", "view"}
                and isinstance(declared, dict)
                and declared["source"] == TABLE_SOURCE
                and declared["metrics"] is not None,
                f"{label} apunta a una vista de tabla sin métricas declaradas",
            )
            _require(sha256(file["path"]) == file["sha256"], f"{label} cambió")
            reports.append(dict(kind=kind, view=view, path=file["path"], sha256=file["sha256"]))
            continue
        _require(
            kind in (WALK_SOURCE, PORTFOLIO_SOURCE) and set(entry) == {"kind", "path", "sha256"},
            f"{label} no es un informe admitido",
        )
        report, report_digest = read_manifest(file["path"], MAX_REPORT_BYTES)
        _require(report_digest == file["sha256"], f"{label} cambió")
        sessions = file["path"].parent / "sessions.parquet"
        _require(
            isinstance(report, dict)
            and report.get("kind") == kind
            and report.get("status") == "completed"
            and report.get("final_test_opened") is False
            and report.get("scope") == scope
            and isinstance(report.get("artifacts"), dict)
            and sessions.is_file()
            and not sessions.is_symlink()
            and sha256(sessions) == report["artifacts"].get("sessions.parquet"),
            f"{label} no es un informe completado de {scope} con su tabla por sesión intacta",
        )
        reports.append(
            dict(
                kind=kind,
                path=file["path"],
                sha256=file["sha256"],
                report=report,
                sessions=sessions,
            )
        )
    published = [item for item in reports if item["kind"] != TABLE_SOURCE]
    _require(published, "Se necesita al menos un informe de comparación o de cartera")
    first = published[0]["report"]
    reference = dict(
        markets=first["markets"],
        edition=first["edition"],
        windows={w: v["view_sha256"] for w, v in first["windows"].items()},
    )
    weighting = config["metrics"]["market_weighting"]
    for item in published:
        report = item["report"]
        _require(
            report["markets"] == reference["markets"]
            and report["edition"] == reference["edition"]
            and {w: v["view_sha256"] for w, v in report["windows"].items()} == reference["windows"],
            f"{item['path']} no comparte mercados, edición y vistas con los demás informes",
        )
        if item["kind"] == WALK_SOURCE:
            _require(
                report["metrics"]["market_weighting"] == weighting,
                f"{item['path']} pondera los mercados de otra forma",
            )
    portfolios = [
        item["report"]["declaration"] for item in published if item["kind"] == PORTFOLIO_SOURCE
    ]
    _require(
        all(declaration == portfolios[0] for declaration in portfolios),
        "Los informes de cartera no comparten la declaración",
    )
    hours = None
    if manifest["hours"] is not None:
        hours_file = walk._file(folder, manifest["hours"], "Las horas")
        hours = load_hours(hours_file["path"], scope)
        _require(hours["sha256"] == hours_file["sha256"], "El documento de horas cambió")
    return dict(
        sha256=digest,
        scope=scope,
        markets=reference["markets"],
        windows=reference["windows"],
        reports=reports,
        hours=hours,
    )


# Series por sesión


def _groups_of(table, keys):
    """Filas de cada combinación de claves, en el orden de su primera aparición."""
    codes = []
    for key in keys:
        column = table.column(key)
        if column.null_count:
            column = column.cast("string").fill_null("\0")
        _, inverse = np.unique(column.to_numpy(zero_copy_only=False), return_inverse=True)
        codes.append(inverse.astype(np.int64))
    combined = np.zeros(table.num_rows, dtype=np.int64)
    for code in codes:
        combined = combined * (int(code.max(initial=0)) + 1) + code
    values, first, inverse = np.unique(combined, return_index=True, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse, minlength=len(values)))
    groups = np.split(order, bounds[:-1])
    return [groups[i] for i in np.argsort(first, kind="stable")]


def _sorted_part(table, rows, markets, label):
    """Filas de un brazo y semilla en orden (mercado, instante), sin sesiones repetidas."""
    part = table.take(rows)
    names = part.column("market").to_numpy(zero_copy_only=False)
    _require(set(names) <= set(markets), f"{label} tiene un mercado fuera del ámbito")
    market = np.array([markets.index(name) for name in names], dtype=np.int64)
    moment = part.column("prediction_at").cast("int64").to_numpy()
    order = np.lexsort((moment, market))
    market, moment = market[order], moment[order]
    repeated = (market[1:] == market[:-1]) & (moment[1:] == moment[:-1])
    _require(not repeated.any(), f"{label} repite una sesión")
    return dict(table=part.take(order), market=market, time=moment)


def _masked(part, name):
    column = part.column(name)
    defined = column.is_valid().to_numpy(zero_copy_only=False)
    values = column.fill_null(0).to_numpy(zero_copy_only=False).astype(np.float64)
    return np.where(defined, values, 0.0), defined


def _ratio(hits, calls):
    defined = calls > 0
    return np.where(defined, hits / np.maximum(calls, 1), 0.0), defined


def _emits_quantiles(part):
    """La tabla une brazos con y sin cuantiles: las columnas nulas no cuentan como emitidas."""
    columns = [name for name in part.column_names if name.startswith("pinball_")]
    return bool(columns) and all(part.column(name).null_count == 0 for name in columns)


def forecast_values(part, metric):
    """Valores y máscara por sesión de la tabla publicada, como ``SessionScores.series``.

    Devuelve None si el brazo no emite lo necesario, por ejemplo cuantiles.
    """
    if quantile_metric(metric) and not _emits_quantiles(part):
        return None
    if metric in ("mae", "mse"):
        values = part.column(metric).to_numpy().astype(np.float64)
        return values, np.ones(len(values), dtype=bool)
    if metric in ("direction_accuracy", "rank_ic", "sign_brier"):
        return _masked(part, metric)
    if metric in ("up_precision", "down_precision"):
        side = metric.split("_")[0]
        return _ratio(
            part.column(f"{side}_hits").to_numpy(), part.column(f"{side}_calls").to_numpy()
        )
    if metric == "pinball":
        columns = [name for name in part.column_names if name.startswith("pinball_")]
        values = np.column_stack([part.column(name).to_numpy() for name in columns]).mean(axis=1)
        return values, np.ones(len(values), dtype=bool)
    _require(metric.startswith(INTERVAL_SCORE), f"Métrica {metric} sin lectura")
    name = f"interval_score_{float(metric.removeprefix(INTERVAL_SCORE)):g}"
    _require(name in part.column_names, f"La tabla no publica {name}")
    values = part.column(name).to_numpy().astype(np.float64)
    return values, np.ones(len(values), dtype=bool)


def _sign_bins(part):
    if not _emits_quantiles(part):
        return None
    return [
        pc.list_flatten(part.column(name)).to_numpy().astype(np.float64).reshape(-1, SIGN_BINS)
        for name in ("sign_bin_rows", "sign_bin_probability", "sign_bin_up")
    ]


def _cohort(label, market, moment, markets):
    digest = hashlib.sha256(b"mars-titan-matrix-cohort-v1\0")
    digest.update(f"{label}\0{'+'.join(markets)}\0".encode())
    digest.update(market.astype("<i8").tobytes())
    digest.update(moment.astype("<i8").tobytes())
    return digest.hexdigest()


def _market_views(markets):
    views = {"+".join(markets): None}
    if len(markets) > 1:
        views.update({market: code for code, market in enumerate(markets)})
    return views


def _selection(seed, code):
    return np.ones(len(seed["market"]), dtype=bool) if code is None else seed["market"] == code


def _periods(moment):
    _, period = np.unique(np.floor_divide(moment, DAY_MICROSECONDS), return_inverse=True)
    return period.astype(np.int64)


def _series(seeds, metric, loss, view_label, code, markets, values_of):
    """Serie del brazo en una vista de mercado: media de las semillas sesión a sesión."""
    items = []
    for seed in seeds:
        mask = _selection(seed, code)
        found = values_of(seed, metric)
        if found is None or not mask.any():
            return None
        values, defined = found
        market, moment = seed["market"][mask], seed["time"][mask]
        items.append(
            SessionSeries(
                metric,
                loss,
                _cohort(view_label, market, moment, markets),
                tuple(markets),
                market,
                _periods(moment),
                np.where(defined[mask], values[mask], 0.0),
                defined[mask].copy(),
            )
        )
    return SessionSeries.average(items)


def _keep(store, arm, seeds, label):
    """Guardar las semillas de un brazo, o exigir que coincidan con las de otro informe."""
    if arm not in store:
        store[arm] = seeds
        return
    previous = store[arm]
    _require(
        len(previous) == len(seeds)
        and all(
            np.array_equal(a["market"], b["market"])
            and np.array_equal(a["time"], b["time"])
            and a["table"].equals(b["table"])
            for a, b in zip(previous, seeds, strict=True)
        ),
        f"{label}: el brazo {arm} difiere entre informes",
    )


def forecast_arms(sources):
    """Semillas por brazo y variante de cuantiles de todos los informes walk-forward."""
    arms = {"raw": {}, "calibrated": {}}
    markets = list(sources["markets"])
    for item in sources["reports"]:
        if item["kind"] != WALK_SOURCE:
            continue
        table = pq.read_table(item["sessions"])
        windows = set(item["report"]["windows"])
        collected = {}
        for rows in _groups_of(table, ("arm", "seed", "quantiles")):
            arm = table.column("arm")[int(rows[0])].as_py()
            variant = table.column("quantiles")[int(rows[0])].as_py()
            part = table.take(rows)
            present = set(part.column("window").to_pylist())
            _require(present <= windows, f"{item['path']}: {arm} tiene ventanas ajenas")
            if present != windows:
                _require(variant == "calibrated", f"{item['path']}: {arm} no cubre las ventanas")
                collected.setdefault((variant, arm), []).append(None)
                continue
            seed = _sorted_part(table, rows, markets, f"{item['path']}: {arm}")
            collected.setdefault((variant, arm), []).append(seed)
        for (variant, arm), seeds in collected.items():
            if None in seeds:
                continue
            _keep(arms[variant], arm, seeds, str(item["path"]))
    return arms


def table_arms(sources, view, metrics):
    """Semillas por brazo y métrica de las tablas por sesión de otra etapa."""
    arms = {}
    markets = list(sources["markets"])
    for item in sources["reports"]:
        if item["kind"] != TABLE_SOURCE or item["view"] != view:
            continue
        table = pq.read_table(item["path"])
        _require(
            set(TABLE_COLUMNS) <= set(table.column_names),
            f"{item['path']} no tiene las columnas {TABLE_COLUMNS}",
        )
        _require(
            set(table.column("metric").to_pylist()) <= set(metrics),
            f"{item['path']} trae métricas no declaradas en la vista {view}",
        )
        collected = {}
        for rows in _groups_of(table, ("metric", "arm", "seed")):
            metric = table.column("metric")[int(rows[0])].as_py()
            arm = table.column("arm")[int(rows[0])].as_py()
            seed = _sorted_part(table, rows, markets, f"{item['path']}: {arm}/{metric}")
            values, defined = _masked(seed["table"], "value")
            _require(np.isfinite(values).all(), f"{item['path']}: {arm}/{metric} no es finita")
            seed["values"] = (values, defined)
            collected.setdefault(arm, {}).setdefault(metric, []).append(seed)
        for arm, by_metric in collected.items():
            for metric, seeds in by_metric.items():
                key = (metric, arm)
                _keep(arms, key, seeds, str(item["path"]))
    return arms


# Contrastes


def _resampling(comparison):
    return dict(
        block_length=comparison["block_length"],
        replicates=comparison["replicates"],
        seed=comparison["seed"],
        confidence=comparison["confidence"],
        sensitivity_block_lengths=tuple(comparison["sensitivity_block_lengths"]),
    )


def _applicable(contrasts, usable):
    ready = {n: c for n, c in contrasts.items() if all(arm in usable for arm in c)}
    skipped = sorted(set(contrasts) - set(ready))
    return ready, skipped


def _with_cost(result, sign, hours):
    if hours is not None:
        for row in result.get("contrasts", []):
            row["cost"] = cost_adjusted(row, sign, hours)
    return result


def ece_contrasts(cells, contrasts, comparison):
    """Contrastes del ECE del signo con las mismas réplicas por días para todos los brazos.

    ``cells`` da por brazo una lista por semilla de (filas, probabilidad, subidas) por día e
    intervalo. Cada réplica calcula el ECE de cada semilla con los días remuestreados y
    promedia las semillas, como la fiabilidad del informe walk-forward. La corrección de la
    familia es el máximo estudentizado de ``compare_series``.
    """
    arms = list(cells)
    estimate = {}
    for arm in arms:
        per_seed = [
            expected_calibration_error(*(cell.sum(axis=0) for cell in seed)) for seed in cells[arm]
        ]
        estimate[arm] = None if None in per_seed else float(np.mean(per_seed))
    names = list(contrasts)
    point = np.array(
        [
            np.nan
            if any(estimate[arm] is None for arm in contrasts[name])
            else math.fsum(c * estimate[arm] for arm, c in contrasts[name].items())
            for name in names
        ]
    )
    periods = cells[arms[0]][0][0].shape[0]
    block, replicates = comparison["block_length"], comparison["replicates"]
    reason, draws = None, None
    if block >= periods:
        reason = "Se necesitan más días que la longitud del bloque"
    elif np.isnan(point).all():
        reason = "Ningún contraste tiene filas con objetivo no nulo en todos sus brazos"
    else:
        rng = np.random.default_rng(comparison["seed"])
        sampled = {arm: [] for arm in arms}
        for offset in range(0, replicates, _ECE_CHUNK):
            size = min(_ECE_CHUNK, replicates - offset)
            counts = circular_block_counts(rng, size, periods, block).astype(np.float64)
            for arm in arms:
                values = [
                    expected_calibration_error(*(counts @ cell for cell in seed))
                    for seed in cells[arm]
                ]
                sampled[arm].append(np.mean(values, axis=0))
        by_arm = {arm: np.concatenate(values) for arm, values in sampled.items()}
        draws = np.column_stack(
            [sum(c * by_arm[arm] for arm, c in contrasts[name].items()) for name in names]
        )
    defined = ~np.isnan(point)
    if draws is not None:
        defined &= ~np.isnan(draws).any(axis=0)
    bounds, critical = None, None
    if draws is not None and defined.any():
        bounds = family_intervals(point[defined], draws[:, defined], comparison["confidence"])
        critical = bounds.pop("critical")
    rows, position = [], 0
    for index, name in enumerate(names):
        row = dict(
            name=name,
            coefficients={arm: float(c) for arm, c in contrasts[name].items()},
            estimate=None if np.isnan(point[index]) else float(point[index]),
            interval=None,
            simultaneous_interval=None,
            bootstrap_standard_error=None,
            simultaneous_excludes_zero=None,
        )
        if bounds is not None and defined[index]:
            row["interval"] = [float(bounds["lower"][position]), float(bounds["upper"][position])]
            joint = [float(bounds["joint_lower"][position]), float(bounds["joint_upper"][position])]
            row["simultaneous_interval"] = joint
            row["bootstrap_standard_error"] = float(bounds["spread"][position])
            row["simultaneous_excludes_zero"] = bool(joint[0] > 0 or joint[1] < 0)
            position += 1
        rows.append(row)
    return dict(
        schema_version=1,
        kind="paired_ece_contrasts",
        metric=SIGN_ECE,
        loss=True,
        resampling=dict(
            method="circular_block_bootstrap",
            unit="utc_calendar_day_with_all_sessions_and_assets",
            periods=periods,
            block_length=block,
            replicates=replicates,
            seed=comparison["seed"],
            reason=reason,
            interval="percentile_marginal_and_max_absolute_studentized",
        ),
        multiplicity=dict(
            method="max_absolute_studentized_bootstrap",
            family_size=int(defined.sum()),
            critical_value=critical,
        ),
        contrasts=rows,
    )


def _ece_cells(seeds, code, label, markets):
    """Celdas por día e intervalo de probabilidad de cada semilla, y la huella de sus sesiones."""
    cells, cohort = [], None
    for seed in seeds:
        mask = _selection(seed, code)
        bins = _sign_bins(seed["table"])
        if bins is None or not mask.any():
            return None, None
        moment = seed["time"][mask]
        periods = _periods(moment)
        per_seed = []
        for values in bins:
            total = np.zeros((int(periods.max()) + 1, SIGN_BINS))
            np.add.at(total, periods, values[mask])
            per_seed.append(total)
        current = _cohort(label, seed["market"][mask], moment, markets)
        _require(cohort in (None, current), "Las semillas no comparten sesiones")
        cohort = current
        cells.append(per_seed)
    return cells, cohort


def forecast_view(config, families, seeds_by_arm, metrics, markets, hours):
    comparison = config["comparison"]
    options = _resampling(comparison)
    weighting = config["metrics"]["market_weighting"]
    used = sorted(
        {a for ready in families.values() for c in ready.values() for a in c if a in seeds_by_arm}
    )
    result = {}
    for label, code in _market_views(markets).items():
        result[label] = {}
        for metric in metrics:
            lower = metric in LOWER_IS_BETTER or metric.startswith(INTERVAL_SCORE)
            sign = _orient(lower)
            values = {}
            if metric == SIGN_ECE:
                cohorts = set()
                for arm in used:
                    cells, cohort = _ece_cells(seeds_by_arm[arm], code, label, markets)
                    if cells is not None:
                        values[arm] = cells
                        cohorts.add(cohort)
                _require(len(cohorts) <= 1, f"Los brazos no comparten sesiones en {label}")
            else:
                for arm in used:
                    found = _series(
                        seeds_by_arm[arm],
                        metric,
                        lower,
                        label,
                        code,
                        markets,
                        lambda seed, name: forecast_values(seed["table"], name),
                    )
                    if found is not None:
                        values[arm] = found
            rows = {}
            for family, ready in families.items():
                contrasts, skipped = _applicable(ready, values)
                if not contrasts:
                    rows[family] = dict(
                        reason="Ningún contraste tiene la métrica en todos sus brazos",
                        not_applicable=skipped,
                    )
                    continue
                arms = {a: values[a] for c in contrasts.values() for a in c}
                if metric == SIGN_ECE:
                    entry = ece_contrasts(arms, contrasts, comparison)
                else:
                    entry = compare_series(arms, contrasts, market_weighting=weighting, **options)
                entry["not_applicable"] = skipped
                rows[family] = _with_cost(entry, sign, hours)
            result[label][metric] = rows
    return result


def table_view(config, families, arms, metrics, markets, hours):
    comparison = config["comparison"]
    options = _resampling(comparison)
    weighting = config["metrics"]["market_weighting"]
    result = {}
    for label, code in _market_views(markets).items():
        result[label] = {}
        for metric, orientation in metrics.items():
            loss = orientation == "loss"
            series = {}
            for (name, arm), seeds in arms.items():
                if name != metric:
                    continue
                found = _series(
                    seeds, metric, loss, label, code, markets, lambda seed, _: seed["values"]
                )
                if found is not None:
                    series[arm] = found
            rows = {}
            for family, ready in families.items():
                contrasts, skipped = _applicable(ready, series)
                if not contrasts:
                    rows[family] = dict(reason="Ningún contraste tiene la métrica en sus brazos")
                    continue
                used = {arm: series[arm] for arm in {a for c in contrasts.values() for a in c}}
                entry = compare_series(used, contrasts, market_weighting=weighting, **options)
                entry["not_applicable"] = skipped
                rows[family] = _with_cost(entry, _orient(loss), hours)
            result[label][metric] = rows
    return result


def portfolio_arms(sources, costs):
    """Rendimientos netos por coste y negociación por brazo y mercado, media de semillas."""
    markets = list(sources["markets"])
    arms = {}
    for item in sources["reports"]:
        if item["kind"] != PORTFOLIO_SOURCE:
            continue
        table = pq.read_table(item["sessions"])
        collected = {}
        for rows in _groups_of(table, ("arm", "seed")):
            arm = table.column("arm")[int(rows[0])].as_py()
            seed = _sorted_part(table, rows, markets, f"{item['path']}: {arm}")
            collected.setdefault(arm, []).append(seed)
        for arm, seeds in collected.items():
            _keep(arms, arm, seeds, str(item["path"]))
    result = {}
    for code, market in enumerate(markets):
        result[market] = {}
        for arm, seeds in arms.items():
            times, net, traded = [], [], []
            for seed in seeds:
                mask = seed["market"] == code
                table = seed["table"].filter(pa.array(mask))
                times.append(seed["time"][mask])
                net.append(
                    np.stack([table.column(f"net_return_{cost}bps").to_numpy() for cost in costs])
                )
                traded.append(table.column("traded").to_numpy().astype(np.float64))
            _require(
                all(np.array_equal(times[0], other) for other in times),
                f"Las semillas de {arm} no comparten sesiones en {market}",
            )
            if len(times[0]):
                result[market][arm] = dict(
                    time=times[0], returns=np.mean(net, axis=0), traded=np.mean(traded, axis=0)
                )
    return result


def portfolio_view(config, families, by_market, declaration, statistics, hours):
    comparison = config["comparison"]
    costs = declaration["cost_bps_per_side"]
    result = {}
    for market, arms in by_market.items():
        used = sorted(
            {a for ready in families.values() for c in ready.values() for a in c if a in arms}
        )
        result[market] = {str(cost): {name: {} for name in statistics} for cost in costs}
        if not used:
            continue
        times = arms[used[0]]["time"]
        _require(
            all(np.array_equal(arms[arm]["time"], times) for arm in used),
            f"Los brazos no comparten sesiones en {market}",
        )
        estimate, draws, reason = {}, {}, None
        for start in range(0, len(used), PORTFOLIO_GROUP):
            group = used[start : start + PORTFOLIO_GROUP]
            part, sampled, reason = long_short.bootstrap(
                np.stack([arms[arm]["returns"] for arm in group], axis=-1),
                np.stack([arms[arm]["traded"] for arm in group], axis=-1),
                sessions_per_year=declaration["annualization_sessions"][market],
                block_length=comparison["block_length"],
                replicates=comparison["replicates"],
                seed=comparison["seed"],
            )
            for name in long_short.STATISTICS:
                estimate.setdefault(name, []).append(part[name])
                if sampled is not None:
                    draws.setdefault(name, []).append(sampled[name])
        estimate = {name: np.concatenate(v, axis=-1) for name, v in estimate.items()}
        draws = None if reason else {name: np.concatenate(v, axis=-1) for name, v in draws.items()}
        for i, cost in enumerate(costs):
            for name in statistics:
                sign = _orient(
                    None
                    if long_short.HIGHER_IS_BETTER[name] is None
                    else not long_short.HIGHER_IS_BETTER[name]
                )
                for family, ready in families.items():
                    contrasts, skipped = _applicable(ready, set(used))
                    if not contrasts:
                        continue
                    entry = long_short.contrasts(
                        estimate[name][i],
                        None if draws is None else draws[name][:, i, :],
                        used,
                        contrasts,
                        comparison["confidence"],
                    )
                    entry["not_applicable"] = skipped
                    entry["resampling_reason"] = reason
                    result[market][str(cost)][name][family] = _with_cost(entry, sign, hours)
    return result
