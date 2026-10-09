"""Informe financiero de la etapa de políticas a partir de sus salidas confirmadas.

Lee `stage.json`, `summary.json` y los recibos de cada trabajo de una o varias salidas de
`simulation.campaign_stage`, comprueba que el resumen concuerda con los recibos y publica
el patrimonio por sesión, las métricas financieras, su incertidumbre y los contrastes de
KLPO con sus controles. No ajusta ni evalúa nada: solo agrega episodios ya confirmados.

Reglas declaradas antes de ver resultados:

- La serie de un brazo en una ventana es su patrimonio a cada cierre. El último valor es la
  liquidación hipotética al último cierre, de modo que cada ventana paga su salida.
- Un brazo aprendido con varias semillas reparte el capital a partes iguales entre ellas.
  Su patrimonio es la media de los patrimonios por semilla y una semilla arruinada vale
  cero desde su ruina. Las métricas de cada semilla se publican aparte.
- Las ventanas se encadenan en orden. La serie completa reúne los retornos por sesión de
  las ventanas usadas, como si el patrimonio liquidado de una ventana pasara a la siguiente.
  Si la serie encadenada se arruina, sus métricas terminan en la ruina.
- Una familia compara, en un ámbito, un mercado, un predictor y un coste, KLPO con todos
  sus controles disponibles. Solo usa las ventanas en que todos sus brazos tienen un
  episodio terminado (completo o arruinado) con los mismos cierres. Las ventanas excluidas
  se publican con su motivo.
- Incertidumbre con el bootstrap circular por bloques de sesiones de
  `evaluation.financial_metrics`, con réplicas, semilla, confianza y longitudes de bloque
  declaradas en la sección `report` de las políticas. Los contrastes son KLPO menos cada
  control, con intervalos simultáneos por máximo estudentizado dentro de cada familia y
  métrica. No hay corrección entre predictores, costes ni métricas. Cada familia responde
  una pregunta separada y el coste principal declarado es el que se interpreta primero.
- El índice chino se calcula con niveles del CSI 300 (`simulation.index_benchmark`) si se
  suministran con su huella, y se marca con su base.
- Una ventana sin evaluación porque una serie del universo termina dentro del tramo se
  publica en `survival`. La sensibilidad con retornos de salida declarada es secundaria.
"""

import argparse
import csv
import io
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation.financial_metrics import (
    CONVENTIONS,
    RESAMPLED,
    block_bootstrap,
    equity_metrics,
)

from . import index_benchmark
from .campaign_stage import RUN_KIND, _digest, _record, summarize
from .policy_plan import MARKET_INDEX, _require, load_stage, plan_stage, scope_windows

REPORT_KIND = "historical_masked_rl_financial_report"
SEED_RULE = "equal_capital_per_seed_mean_nav"
WINDOW_RULE = "liquidated_last_close_then_chained_windows"
DIFFERENCE = "primary_minus_control"
SERIES_ENDS = "series_ends_in_tape"
STATUSES = ("completed", "paused")
_RECEIPT_BYTES = 64 * 1024**2


def read_output(stage, output):
    """Resumen y recibos confirmados de una salida de la etapa, con sus comprobaciones."""
    output = Path(output)
    marker, marker_sha = read_manifest(output / "stage.json", 8 * 1024**2)
    summary, summary_sha = read_manifest(output / "summary.json", 64 * 1024**2)
    _require(
        marker.get("kind") == RUN_KIND
        and marker.get("stage_sha256") == stage["sha256"]
        and marker.get("policies_sha256") == stage["policies"]["sha256"]
        and marker.get("final_test_opened") is False
        and summary.get("kind") == RUN_KIND
        and summary.get("identity_sha256") == _digest(marker)
        and summary.get("status") in STATUSES
        and summary.get("final_test_opened") is False,
        f"La salida {output} no pertenece a esta etapa o no está confirmada",
    )
    jobs = plan_stage(stage)
    _require(
        set(summary["jobs"]) == {job["id"] for job in jobs},
        f"El resumen de {output} no enumera el plan de la etapa",
    )
    costs = stage["policies"]["evaluation_costs_bps"]
    receipts = {}
    for job in jobs:
        path = output / "jobs" / job["id"] / "receipt.json"
        if not summary["jobs"][job["id"]]:
            continue
        receipt, digest = read_manifest(path, _RECEIPT_BYTES)
        identity = receipt.get("identity", {})
        _require(
            receipt.get("status") == "completed"
            and receipt.get("final_test_opened") is False
            and all(identity.get(key) == job[key] for key in ("id", "arm", "seed", "window"))
            and isinstance(receipt.get("evaluation"), list)
            and len(receipt["evaluation"]) == len(costs),
            f"El recibo de {job['id']} no corresponde a su trabajo",
        )
        for record, cost in zip(receipt["evaluation"], costs, strict=True):
            _record(record, cost, None)
        receipts[job["id"]] = dict(receipt, sha256=digest, job=job)
    # El resumen publicado debe salir de los mismos recibos.
    _require(
        json.loads(json.dumps(summarize(stage, receipts))) == summary["metrics"],
        f"Las métricas del resumen de {output} no concuerdan con sus recibos",
    )
    return dict(
        path=str(output.resolve()),
        status=summary["status"],
        stage_sha256=marker_sha,
        summary_sha256=summary_sha,
        planned=summary["planned"],
        completed=summary["completed"],
        receipts=receipts,
    )


def _window_nav(record, close_times, capital):
    """Patrimonio a cada cierre de la ventana con la liquidación en el último."""
    equity = record["equity"]
    nav = [float(value) for value in equity["nav"]]
    times = equity["close_times"]
    _require(
        times == close_times[: len(times)] and nav[0] == capital,
        "Los brazos de una ventana deben empezar con el capital y compartir sus cierres",
    )
    if record["status"] == "ruined":
        # Sin patrimonio desde la ruina hasta el final de la ventana.
        return np.array(nav + [0.0] * (len(close_times) - len(nav)))
    _require(len(times) == len(close_times), "Un episodio completo recorre toda la ventana")
    nav[-1] = capital * (1 + record["liquidated_net_return"])
    return np.array(nav)


def _arm_window(records, close_times, capital):
    """Patrimonio de un brazo en una ventana: media de sus semillas, cada una con su capital."""
    navs = [_window_nav(record, close_times, capital) for record in records]
    return dict(
        nav=np.mean(navs, axis=0),
        seeds=navs,
        turnover=math.fsum(record["turnover"] for record in records) / len(records),
        costs=math.fsum(record["costs"] for record in records) / len(records) / capital,
    )


def _returns(nav):
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.where(nav[:-1] > 0, nav[1:] / nav[:-1] - 1, 0.0)
    return returns


def _chained(pieces, capital):
    """Métricas de las ventanas encadenadas, con giro y costes sumados sobre el capital.

    Una ruina encadenada termina la serie de métricas, porque las ventanas siguientes ya no
    tendrían capital. Los retornos completos se conservan para el bootstrap pareado.
    """
    returns = np.concatenate([_returns(piece["nav"]) for piece in pieces])
    growth = np.concatenate(([1.0], np.cumprod(1 + returns)))
    ruin = np.flatnonzero(growth == 0)
    nav = capital * (growth if len(ruin) == 0 else growth[: ruin[0] + 1])
    turnover = math.fsum(piece["turnover"] for piece in pieces)
    costs = math.fsum(piece["costs"] for piece in pieces)
    return returns, nav, equity_metrics(nav, turnover=turnover, costs=costs)


def _flip(bootstrap):
    """Expresar las diferencias como KLPO menos control. Los intervalos son simétricos."""

    def negate(pair):
        return None if pair is None else [-pair[1], -pair[0]]

    for entry in (bootstrap, *bootstrap["sensitivity"]):
        for rows in entry["differences"].values():
            for row in rows.values():
                if row["estimate"] is not None:
                    row["estimate"] = -row["estimate"]
                row["interval"] = negate(row["interval"])
                row["simultaneous_interval"] = negate(row["simultaneous_interval"])
    bootstrap["difference"] = DIFFERENCE
    return bootstrap


def _benchmark_record(levels, close_times, market, capital, cost):
    record, reason = index_benchmark.index_episode(
        levels, close_times, market=market, capital=capital, cost_bps=cost
    )
    if record is None:
        return None, reason
    return dict(record, status="completed", cost_bps=cost, reason=None), None


def families(stage, output, benchmarks):
    """Familias pareadas de una salida por ámbito, mercado, predictor y coste."""
    policies = stage["policies"]
    capital = float(policies["environment"]["capital"])
    primary = policies["contrasts"]["primary"]
    controls = policies["contrasts"]["controls"]
    records = {}
    for receipt in output["receipts"].values():
        job = receipt["job"]
        for record in receipt["evaluation"]:
            key = (job["scope"], job["market"], job["predictor"], record["cost_bps"])
            records.setdefault(key, {}).setdefault(job["window"], {}).setdefault(
                job["arm"], []
            ).append(record)
    planned = Counter(
        (job["scope"], job["market"], job["predictor"], job["window"], job["arm"])
        for job in plan_stage(stage)
    )
    result = []
    for scope in stage["scopes"]:
        windows = [row["window"] for row in scope_windows(stage, scope)]
        markets = stage["campaign"]["comparison_config"]["resolved_scopes"][scope]["markets"]
        for market in markets:
            for predictor in stage["predictors"]:
                arms = [
                    arm
                    for arm in (primary, *controls)
                    if any(planned[(scope, market, predictor, w, arm)] for w in windows)
                ]
                benchmark = benchmarks.get(market) if MARKET_INDEX not in arms else None
                if benchmark is not None:
                    arms.append(MARKET_INDEX)
                for cost in policies["evaluation_costs_bps"]:
                    family = _family(
                        records.get((scope, market, predictor, cost), {}),
                        windows,
                        arms,
                        planned=lambda w, a, key=(scope, market, predictor): planned[(*key, w, a)],
                        benchmark=benchmark,
                        market=market,
                        capital=capital,
                        cost=cost,
                    )
                    family.update(scope=scope, market=market, predictor=predictor, cost_bps=cost)
                    result.append(family)
    return result


def _family(found, windows, arms, *, planned, benchmark, market, capital, cost):
    pieces = {arm: [] for arm in arms}
    used, excluded = [], {}
    for window in windows:
        present = found.get(window, {})
        missing = [arm for arm in arms if len(present.get(arm, [])) < planned(window, arm)]
        if missing:
            excluded[window] = dict(reason="pending_jobs", arms=missing)
            continue
        failed = {
            arm: sorted({r["reason"] for r in present.get(arm, []) if r["status"] == "failed"})
            for arm in arms
        }
        failed = {arm: reasons for arm, reasons in failed.items() if reasons}
        if failed:
            excluded[window] = dict(reason="failed_episodes", arms=failed)
            continue
        complete = [r for arm in arms for r in present.get(arm, []) if r["status"] == "completed"]
        if not complete:
            excluded[window] = dict(reason="every_arm_ruined")
            continue
        close_times = complete[0]["equity"]["close_times"]
        if benchmark is not None:
            record, reason = _benchmark_record(benchmark, close_times, market, capital, cost)
            if record is None:
                excluded[window] = dict(reason=reason, arms=[MARKET_INDEX])
                continue
            present = dict(present, **{MARKET_INDEX: [record]})
        for arm in arms:
            piece = _arm_window(present[arm], close_times, capital)
            pieces[arm].append(dict(piece, window=window, close_times=close_times))
        used.append(window)
    family = dict(windows=used, excluded=excluded, arms={}, equity={}, bootstrap=None)
    if not used:
        return family
    returns = {}
    for arm in arms:
        series, nav, metrics = _chained(pieces[arm], capital)
        seeds = len(pieces[arm][0]["seeds"])
        per_seed = None
        if seeds > 1:
            per_seed = []
            for index in range(seeds):
                chained = [dict(p, nav=p["seeds"][index]) for p in pieces[arm]]
                per_seed.append(_chained(chained, capital)[2])
        family["arms"][arm] = dict(
            metrics,
            seeds=seeds,
            per_seed=per_seed,
            basis=index_benchmark.BASIS
            if benchmark is not None and arm == MARKET_INDEX
            else "engine_close_valuation",
        )
        family["equity"][arm] = dict(
            close_times=[t for p in pieces[arm] for t in p["close_times"]],
            windows=[p["window"] for p in pieces[arm] for _ in p["close_times"]],
            nav=[float(v) for p in pieces[arm] for v in p["nav"]],
        )
        returns[arm] = series
    return dict(family, returns=returns)


def survival(policies, output):
    """Ventanas sin evaluación porque una serie del universo termina dentro del tramo.

    La regla principal excluye esas ventanas para todos los brazos. La sensibilidad
    declarada, con retornos de salida, es secundaria y queda pendiente si hay alguna.
    """
    affected = {}
    for receipt in output["receipts"].values():
        failure = receipt["identity"]["tapes"]["failure"]
        if not failure or failure.get("reason") != "universe_assets_excluded":
            continue
        ended = {a for a, reason in failure["excluded"].items() if reason == SERIES_ENDS}
        if ended:
            job = receipt["job"]
            key = (job["scope"], job["market"], job["window"])
            affected[key] = affected.get(key, set()) | ended
    return dict(
        policies["survival_sensitivity"],
        status="secondary_evaluation_pending" if affected else "no_affected_windows",
        affected=[
            dict(scope=scope, market=market, window=window, assets=sorted(assets))
            for (scope, market, window), assets in sorted(affected.items())
        ],
    )


def _bootstrap(family, report, primary):
    returns = family.pop("returns", None)
    if returns is None or len(returns) < 2:
        family["bootstrap"] = dict(reason="Una familia necesita KLPO y algún control con datos")
        return family
    sessions = len(next(iter(returns.values())))
    if sessions <= report["block_length"]:
        family["bootstrap"] = dict(reason="Se necesitan más sesiones que la longitud del bloque")
        return family
    family["bootstrap"] = _flip(
        block_bootstrap(
            returns,
            base=primary,
            block_length=report["block_length"],
            replicates=report["replicates"],
            seed=report["seed"],
            confidence=report["confidence"],
            sensitivity=tuple(report["block_length_sensitivity"]),
        )
    )
    return family


METRIC_COLUMNS = (
    "sessions",
    "cumulative_return_percent",
    "annualized_return_percent",
    "volatility",
    "sharpe",
    "sortino",
    "max_drawdown",
    "turnover",
    "costs",
    "ruined",
)


def _write_text(path, text):
    safe_destination(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _csv(rows, columns):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _tables(sections):
    metrics, contrasts, equity = [], [], []
    for section in sections:
        for family in section["families"]:
            key = dict(
                output=section["output"]["path"],
                scope=family["scope"],
                market=family["market"],
                predictor=family["predictor"],
                cost_bps=family["cost_bps"],
            )
            for arm, values in family["arms"].items():
                metrics.append(
                    dict(key, arm=arm, windows=len(family["windows"]))
                    | {name: values[name] for name in METRIC_COLUMNS}
                )
            for arm, series in family.pop("equity").items():
                for window, time, nav in zip(
                    series["windows"], series["close_times"], series["nav"], strict=True
                ):
                    equity.append(dict(key, arm=arm, window=window, close_time=time, nav=nav))
            bootstrap = family["bootstrap"]
            if not bootstrap or "differences" not in bootstrap:
                continue
            for metric in RESAMPLED:
                for arm, row in bootstrap["differences"][metric].items():
                    contrasts.append(
                        dict(key, metric=metric, primary=bootstrap["base"], control=arm)
                        | dict(
                            estimate=row["estimate"],
                            lower=None if row["interval"] is None else row["interval"][0],
                            upper=None if row["interval"] is None else row["interval"][1],
                            simultaneous_lower=None
                            if row["simultaneous_interval"] is None
                            else row["simultaneous_interval"][0],
                            simultaneous_upper=None
                            if row["simultaneous_interval"] is None
                            else row["simultaneous_interval"][1],
                            simultaneous_excludes_zero=row["simultaneous_excludes_zero"],
                        )
                    )
    return metrics, contrasts, equity


def build_report(stage_path, outputs, destination, *, benchmarks=None):
    """Leer las salidas, calcular familias y escribir el informe, sus tablas y el patrimonio.

    `benchmarks` asigna a un mercado la ruta y la huella de los niveles de su índice.
    """
    stage = load_stage(stage_path)
    policies = stage["policies"]
    report = policies["report"]
    declared = report["benchmarks"]
    benchmarks = dict(benchmarks or {})
    _require(
        set(benchmarks) <= set(declared),
        "Solo se admiten los benchmarks declarados en las políticas",
    )
    levels = {
        market: index_benchmark.read_levels(Path(path), digest)
        for market, (path, digest) in benchmarks.items()
    }
    destination = Path(destination)
    sections = []
    for path in outputs:
        output = read_output(stage, path)
        found = [
            _bootstrap(family, report, policies["contrasts"]["primary"])
            for family in families(stage, output, levels)
        ]
        exits = survival(policies, output)
        output.pop("receipts")
        sections.append(dict(output=output, families=found, survival=exits))
    metrics, contrasts, equity = _tables(sections)
    result = dict(
        schema_version=1,
        kind=REPORT_KIND,
        stage_sha256=stage["sha256"],
        policies_sha256=policies["sha256"],
        declared=dict(report=report, contrasts=policies["contrasts"]),
        conventions=dict(CONVENTIONS, seeds=SEED_RULE, windows=WINDOW_RULE, difference=DIFFERENCE),
        benchmarks={
            market: dict(
                name=declared[market],
                basis=index_benchmark.BASIS,
                sha256=levels[market]["sha256"],
            )
            for market in levels
        },
        missing_benchmarks=sorted(set(declared) - set(levels)),
        sections=sections,
        final_test_opened=False,
    )
    atomic_json(destination / "report.json", result)
    _write_text(destination / "metrics.csv", _csv(metrics, list(metrics[0]) if metrics else []))
    columns = list(contrasts[0]) if contrasts else []
    _write_text(destination / "contrasts.csv", _csv(contrasts, columns))
    table = pa.Table.from_pylist(equity) if equity else pa.table({})
    temporary = destination / ".equity.parquet.tmp"
    pq.write_table(table, temporary)
    os.replace(temporary, destination / "equity.parquet")
    return result


def _benchmark_argument(values):
    result = {}
    for market, path, digest in values or ():
        _require(market not in result, f"El benchmark de {market} se repite")
        result[market] = (path, digest)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", action="append", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument(
        "--benchmark",
        nargs=3,
        action="append",
        metavar=("MERCADO", "NIVELES", "SHA256"),
        help="Niveles del índice declarado de un mercado, con su huella",
    )
    args = parser.parse_args(argv)
    result = build_report(
        args.stage, args.output, args.report, benchmarks=_benchmark_argument(args.benchmark)
    )
    families = [family for section in result["sections"] for family in section["families"]]
    print(
        json.dumps(
            dict(
                report=str(args.report / "report.json"),
                families=len(families),
                with_bootstrap=sum("differences" in (f["bootstrap"] or {}) for f in families),
                missing_benchmarks=result["missing_benchmarks"],
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
