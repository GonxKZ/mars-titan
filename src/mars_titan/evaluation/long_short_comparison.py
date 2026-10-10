"""Comparar la cartera larga y corta por cuartiles de los brazos de una comparación walk-forward.

Lee la configuración declarada (versión 4 o posterior, sección ``long_short``) y el mismo
manifiesto de fuentes que ``walk_forward_comparison``, con sus comprobaciones: política,
vistas, ventanas, tramo de evaluación, reserva de 2024 cerrada y mismas filas y objetivos
en todos los brazos. Las predicciones se leen con ``_read_predictions``, la única lectura
por fila. Los precios proceden de la edición sin ajustar (``simulation.session_prices``).
Con el diseño conjunto de la versión 5 se aplican las mismas exclusiones que en la
comparación: un mercado solo entra en las ventanas en las que es elegible y un brazo
prestado del modelo conjunto conserva solo las filas del mercado de su ámbito.

Cada ventana y brazo produce un libro por sesión. Las sesiones de todas las ventanas se
unen y cada mercado se informa por separado, porque una cartera no mezcla monedas ni
calendarios. Las semillas de un brazo se promedian sesión a sesión antes de calcular
estadísticos y remuestrear, y cada semilla conserva además su resumen. La incertidumbre
usa el bootstrap circular por bloques y las familias de contrastes de la comparación, con
la corrección por máximo estudentizado dentro de cada familia, coste y estadístico.

Con ``--aggregates`` los libros de cada ventana se leen de los agregados que guardó la
retención v2 (``window_aggregates.write_long_short``) en lugar de las predicciones por fila.
El informe sale idéntico, porque todo lo posterior a cada ventana parte de esos libros.
"""

import argparse
import json
import resource
import time
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import safe_destination
from mars_titan.data.input_policy import policy_identity
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.evaluation import long_short
from mars_titan.evaluation import walk_forward_comparison as walk
from mars_titan.simulation.session_prices import REASONS, SessionPrices

REPORT_KIND = "long_short_comparison_report"
SECTION = "long_short"
_EXECUTION = (
    "open",
    "close",
    "long_entry",
    "short_entry",
    "blocked_long",
    "blocked_short",
    "exit_limit_long",
    "exit_limit_short",
    "buy_tax",
    "sell_tax",
    "reason",
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _execution(prices, panel, assets):
    """Ejecuciones de cada fila del panel en su orden canónico, mercado a mercado."""
    result = None
    for code, market in enumerate(panel.markets):
        mask = panel.market == code
        if mask.any():
            local = prices[market].executions(assets[mask], panel.prediction_at[mask])
            if result is None:
                result = {
                    name: np.zeros(panel.rows, dtype=local[name].dtype) for name in _EXECUTION
                }
            for name in _EXECUTION:
                result[name][mask] = local[name]
    return result


def _ordered(arms):
    """Brazos con predicciones primero y el control cero al final, como en la comparación."""
    return sorted(arms.items(), key=lambda item: item[1]["output"] == walk.ZERO_CONTROL)


def _window(sources, config, window_id, edition, declared):
    """Libros por sesión de todos los brazos y semillas en una ventana, con las mismas filas."""
    window = sources["windows"][window_id]
    start, end = window["evaluation"]
    markets = [market for market in sources["markets"] if window_id in sources["eligible"][market]]
    prices = {market: SessionPrices(edition, market, start, end) for market in markets}
    options = {key: declared[key] for key in ("fraction", "min_assets", "exposure")}
    books, unfilled, reference, targets, execution = {}, {}, None, None, None
    for name, arm in _ordered(config["arms"]):
        for seed in arm["seeds"] or [None]:
            label = f"{name} semilla {seed} en {window_id}"
            if arm["output"] == walk.ZERO_CONTROL:
                table, prediction = targets, np.zeros(targets.num_rows)
            else:
                record = sources["files"][name, seed, window_id]["evaluation"]
                table, prediction = walk._read_predictions(record, walk.COLUMNS), None
                table = walk._restricted(table, sources, window_id, name)[0]
            panel = walk._panel(
                table,
                window,
                "evaluation",
                sources,
                label,
                output=walk.POINT,
                prediction=prediction,
            )
            if reference is None:
                reference = (label, panel)
                targets = table.select(["asset_id", "market", "prediction_at", "target"])
                positions = walk._panel(
                    table,
                    window,
                    "evaluation",
                    sources,
                    label,
                    output=walk.POINT,
                    prediction=np.arange(table.num_rows, dtype=np.float64),
                ).prediction.astype(np.int64)
                assets = table["asset_id"].to_numpy(zero_copy_only=False)[positions]
                execution = _execution(prices, panel, assets)
            walk._same_rows(reference, panel, label)
            weights = long_short.quartile_weights(panel, **options)
            books[name, seed] = long_short.session_book(panel, weights, execution)
            selected = weights != 0
            unfilled[name, seed] = dict(
                Counter(np.asarray(REASONS)[execution["reason"][selected]].tolist()),
                limit_up=int(np.sum((weights > 0) & execution["blocked_long"])),
                limit_down=int(np.sum((weights < 0) & execution["blocked_short"])),
            )
    panel = reference[1]
    rows = Counter(np.asarray(REASONS)[execution["reason"]].tolist())
    sessions = dict(market=panel.session_market, time=panel.session_time)
    identity = {market: value.identity() for market, value in prices.items()}
    return books, unfilled, sessions, dict(rows=dict(rows), prices=identity)


def _join(parts, key):
    return np.concatenate([part[key] for part in parts])


def _finite(value):
    """Número del informe, o None si el estadístico no está definido."""
    return float(value) if np.isfinite(value) else None


def _exposure(book):
    selected = book["selected_long"] + book["selected_short"]
    filled = book["filled_long"] + book["filled_short"]
    return dict(
        mean_long_exposure=float(book["long_exposure"].mean()),
        mean_short_exposure=float(book["short_exposure"].mean()),
        mean_gross_exposure=float((book["long_exposure"] + book["short_exposure"]).mean()),
        mean_cash_share=float(1 - (book["long_exposure"] + book["short_exposure"]).mean()),
        mean_positions=float(filled.mean()),
        trades=int(2 * filled.sum()),
        fill_rate=float(filled.sum() / selected.sum()) if selected.sum() else None,
        exits_at_limit=int(book["exits_at_limit"].sum()),
        sessions_without_positions=int(np.sum(filled == 0)),
    )


def _view(config, declared, keys, books, mask, market):
    """Estadísticos, intervalos y contrastes de un mercado con las semillas promediadas."""
    comparison = config["comparison"]
    costs = declared["cost_bps_per_side"]
    annual = declared["annualization_sessions"][market]
    arms = list(dict.fromkeys(arm for arm, _ in keys))
    by_arm = {arm: [key for key in keys if key[0] == arm] for arm in arms}
    local = {key: {name: values[mask] for name, values in books[key].items()} for key in keys}
    net = {key: long_short.net_returns(local[key], costs) for key in keys}
    returns = np.stack([np.mean([net[k] for k in by_arm[a]], axis=0) for a in arms], axis=-1)
    traded = np.stack([np.mean([local[k]["traded"] for k in by_arm[a]], axis=0) for a in arms], 1)
    estimate, draws, reason = long_short.bootstrap(
        returns,
        traded,
        sessions_per_year=annual,
        block_length=comparison["block_length"],
        replicates=comparison["replicates"],
        seed=comparison["seed"],
    )
    summaries = {}
    for column, arm in enumerate(arms):
        seeds = {}
        for key in by_arm[arm]:
            stats = long_short.statistics(
                net[key][..., None], local[key]["traded"][:, None], annual
            )
            seeds["deterministic" if key[1] is None else str(key[1])] = dict(
                costs={
                    str(cost): {name: _finite(v[i, 0]) for name, v in stats.items()}
                    for i, cost in enumerate(costs)
                },
                execution=_exposure(local[key]),
            )
        summaries[arm] = dict(
            seeds=seeds,
            seed_mean={
                str(cost): {
                    name: long_short.level_intervals(
                        estimate[name][i, column : column + 1],
                        None if draws is None else draws[name][:, i, column : column + 1],
                        comparison["confidence"],
                    )[0]
                    for name in long_short.STATISTICS
                }
                for i, cost in enumerate(costs)
            },
        )
    contrasts = {}
    for i, cost in enumerate(costs):
        contrasts[str(cost)] = {
            name: {
                family: long_short.contrasts(
                    estimate[name][i],
                    None if draws is None else draws[name][:, i, :],
                    arms,
                    coefficients,
                    comparison["confidence"],
                )
                for family, coefficients in config["resolved_families"].items()
            }
            for name in long_short.STATISTICS
        }
    resampling = dict(
        method="circular_block_bootstrap",
        unit="market_session_in_calendar_order",
        sessions=int(mask.sum()),
        block_length=comparison["block_length"],
        replicates=comparison["replicates"],
        seed=comparison["seed"],
        bit_generator="PCG64",
        confidence=comparison["confidence"],
        reason=reason,
    )
    return dict(
        sessions_per_year=annual, resampling=resampling, arms=summaries, contrasts=contrasts
    )


def evaluate_long_short(config_path, sources_path, scope, edition, *, aggregates=None):
    """Calcular el informe y la tabla por sesión de la cartera sin escribir nada.

    `config_path` es la ruta de la configuración o una configuración ya validada.
    `aggregates` es la carpeta de agregados por ventana de la retención v2. Con ella no se
    lee ninguna predicción por fila y cada ventana exige agregados de estas mismas fuentes,
    este código y esta edición de precios.
    """
    started = time.perf_counter()
    config = walk.resolve_config(config_path)
    _require(SECTION in config, "La configuración no declara la cartera larga y corta")
    declared = config[SECTION]
    sources = walk.load_sources(sources_path, config, scope)
    # Desde aquí, los brazos y las familias son los del ámbito evaluado.
    config = walk.scope_config(config, scope)
    parts, sessions, unfilled, windows = {}, [], Counter(), {}
    for window_id in sources["windows"]:
        if aggregates is None:
            computed = _window(sources, config, window_id, edition, declared)
        else:
            from . import window_aggregates

            computed = window_aggregates.read_long_short(
                aggregates, config, sources, window_id, edition
            )
        books, missing, moments, record = computed
        for key, book in books.items():
            parts.setdefault(key, []).append(book)
            for reason, count in missing[key].items():
                unfilled[key, reason] += count
        sessions.append(moments)
        windows[window_id] = dict(
            evaluation=sources["windows"][window_id]["evaluation"],
            view_sha256=sources["views"][window_id],
            rows_by_execution=record["rows"],
        )
        prices = record["prices"]
    keys = list(parts)
    books = {key: {name: _join(parts[key], name) for name in parts[key][0]} for key in keys}
    market_codes, times = _join(sessions, "market"), _join(sessions, "time")
    views, tables = {}, []
    for code, market in enumerate(sources["markets"]):
        mask = market_codes == code
        order = np.argsort(times[mask], kind="stable")
        _require(np.all(np.diff(times[mask][order]) > 0), "Hay sesiones repetidas en un mercado")
        _require(
            np.array_equal(order, np.arange(len(order))),
            "Las ventanas no están en orden cronológico",
        )
        views[market] = _view(config, declared, keys, books, mask, market)
    costs = declared["cost_bps_per_side"]
    for key in keys:
        net = long_short.net_returns(books[key], costs)
        columns = dict(
            arm=pa.array([key[0]] * len(times)),
            seed=pa.array([key[1]] * len(times), pa.int64()),
            market=pa.array(np.asarray(sources["markets"])[market_codes]),
            prediction_at=pa.array(times, pa.timestamp("us", tz="UTC")),
            **books[key],
            **{f"net_return_{cost}bps": net[i] for i, cost in enumerate(costs)},
        )
        tables.append(pa.table(columns))
    report = dict(
        schema_version=1,
        kind=REPORT_KIND,
        status="completed",
        created_at_utc=datetime.now(UTC).isoformat(),
        final_test_opened=False,
        scope=scope,
        markets=sources["markets"],
        configuration=dict(name=config["name"], sha256=config["sha256"]),
        sources_sha256=sources["sha256"],
        **(policy_identity(config["input_policy"]) or dict(input_policy=config["input_policy"])),
        edition=sources["edition"],
        prices=prices,
        declaration=declared,
        assumptions=long_short.ASSUMPTIONS,
        windows=windows,
        unfilled_selected_rows={
            f"{arm}/{'deterministic' if seed is None else seed}": {
                reason: count for (k, reason), count in unfilled.items() if k == (arm, seed)
            }
            for arm, seed in keys
        },
        views=views,
        versions={name: version(name) for name in ("numpy", "pyarrow")},
        analysis_source_sha256={
            name: sha256(Path(__file__).parents[1] / name)
            for name in (
                "evaluation/financial_conventions.py",
                "evaluation/long_short.py",
                "evaluation/long_short_comparison.py",
                "evaluation/paired_comparisons.py",
                "evaluation/walk_forward_comparison.py",
                "simulation/session_prices.py",
                "simulation/market_rules.py",
            )
        },
        resources=dict(
            process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * 1024,
            gpu_used=False,
            elapsed_seconds=time.perf_counter() - started,
        ),
    )
    return report, pa.concat_tables(tables)


def write_long_short(config_path, sources_path, scope, edition, output, *, aggregates=None):
    """Publicar el informe y las sesiones en un directorio nuevo fuera de las fuentes."""
    output = Path(output)
    safe_destination(output)
    _require(not output.exists(), "La salida debe ser nueva")
    for source in (Path(config_path).parent, Path(sources_path).parent, Path(edition)):
        outside_source(source, output)
        outside_source(output, source)
    report, sessions = evaluate_long_short(
        config_path, sources_path, scope, edition, aggregates=aggregates
    )
    json.dumps(report, allow_nan=False)
    output.mkdir(parents=True)
    pq.write_table(sessions, output / "sessions.parquet", compression="zstd")
    report["artifacts"] = {"sessions.parquet": sha256(output / "sessions.parquet")}
    atomic_json(output / "long_short.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--scope", choices=tuple(walk.SCOPES), required=True)
    parser.add_argument("--edition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregates", type=Path, help="Agregados por ventana de la retención v2")
    args = parser.parse_args(argv)
    report = write_long_short(
        args.config,
        args.sources,
        args.scope,
        args.edition,
        args.output,
        aggregates=args.aggregates,
    )
    print(
        f"Cartera de {len(report['views'][report['markets'][0]]['arms'])} brazos en "
        f"{len(report['windows'])} ventanas. Reserva final cerrada."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
