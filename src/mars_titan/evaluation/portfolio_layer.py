# ruff: noqa: E501
"""P, la capa de cartera para las medidas netas sobre la puntuación operable de S2.

Encargo del coordinador del 23 de noviembre de 2026, con el protocolo del programa (§5.12) y la
configuración fijada en `configs/experiments/posttraining-portfolio-layer.json`, cuya huella exige
la orden. Es descriptivo y secundario.

- **Puntuación y universo.** La operable de S2 (k·p + α), con las entradas y la configuración de
  S2, sin las filas con |α| sobre el tope de F. La lectura principal es el tercio más líquido de
  cada sesión; todos los nombres van como secundaria.
- **Libro.** Largo del decil alto y corto del bajo del rango suavizado, con pesos iguales por pata,
  neutral en dinero y exposición bruta 2. El rendimiento es y + α.
- **Suavizado.** s̃ₜ = λ·zₜ + (1 − λ)·s̃ₜ₋₁ por activo sobre el rango tipificado de la sesión; un
  activo que no estaba en la sesión anterior empieza en zₜ.
- **Banda.** Un nombre entra en el decil extremo y solo sale si abandona el extremo de salida.
- **Elección.** (λ, salida) se elige en calibración por el neto medio a 3 pb por lado; a igualdad,
  el más sencillo (λ mayor y, después, salida menor).
- **Costes.** c·Σ|wₜ − wₜ₋₁| por sesión, con c en pb por lado.
- **Control y referencia.** El control es la señal diaria sin suavizar ni banda; la referencia es el
  libro del alfa solo.
- **Medidas en evaluación** por coste: neto, bruto, rotación, Sharpe anualizado, caída máxima y
  coste de equilibrio. Por horizonte (1, 5 y 21 sesiones, en ventanas sin solapar), la fracción
  con neto > 0 (Wilson, y por bloques en las sesiones sueltas) y su neto medio. También la
  diferencia capa − control por sesión, con t de Newey-West de la criba.
- **Regla en el tercio líquido.** La capa entra si su neto a 3 pb supera al control con t ≥ 2 y su
  neto a 5 pb es > 0 con t ≥ 2.

Cada tramo empieza sin cartera y con el suavizado reiniciado. Una sesión con menos nombres que
`min_assets` en el universo no tiene posiciones. Una pata vacía deja la sesión sin posiciones.
Límites declarados: ejecución al precio de la decisión, coste fijo por lado sin impacto, y sin
modelar la disponibilidad de cortos. Nada se elige en evaluación.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata

from mars_titan.evaluation import agent_core as ac
from mars_titan.evaluation import online_aggregator as oa
from mars_titan.evaluation import organ_aggregation as og
from mars_titan.evaluation import posttraining_pilots as pilots
from mars_titan.evaluation import reliability as rel

KIND = "posttraining_portfolio_layer_v1"
# Huella canónica de configs/experiments/posttraining-portfolio-layer.json, la declaración de P.
REGISTERED_SHA256 = "828ee84c022a53f405585cb8521244210e5393c665b182ee01064d35abb0624d"
SPLITS, AGGREGATED = og.SPLITS, og.AGGREGATED


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def checked_declaration(document):
    """La declaración registrada de P y su huella, o un error que dice por qué no lo es."""
    _require(isinstance(document, dict), "La declaración de P es un objeto JSON")
    digest = hashlib.sha256(pilots._canonical(document).encode()).hexdigest()
    _require(
        digest == REGISTERED_SHA256,
        f"La declaración no es la registrada para P (huella {digest[:12]}, registrada "
        f"{REGISTERED_SHA256[:12]}): el protocolo no cambia",
    )
    grid = document["aggregator"]
    _require(
        tuple(grid["lambdas"]) == tuple(oa.RIDGE_LAMBDAS)
        and tuple(grid["half_lives"]) == tuple(oa.RIDGE_HALF_LIVES),
        "El agregador del código no usa la rejilla de ridge_cov_active declarada",
    )
    return document, digest


# ---------------------------------------------------------------------------------------------
# Piezas puras


def session_z(score):
    """Rango tipificado dentro de la sesión: media 0 y desviación 1 (0 si no hay orden)."""
    ranks = rankdata(score)
    spread = ranks.std()
    return (ranks - ranks.mean()) / spread if spread > 0 else np.zeros(len(ranks))


def run_book(sessions, assets, *, smoothing, entry, exit_tail, min_assets):
    """Libro diario con suavizado y banda sobre unas sesiones en el orden del tiempo.

    `sessions` son (clave, índices de activo, puntuación, rendimiento) del universo de cada
    sesión, y `assets` el número total de activos. Devuelve por sesión la clave, el bruto, la
    rotación Σ|wₜ − wₜ₋₁| y los nombres largos y cortos.
    """
    previous_smooth = np.full(assets, np.nan)
    previous_weights = np.zeros(assets)
    result = []
    for key, index, score, returns in sessions:
        weights = np.zeros(assets)
        smooth_now = np.full(assets, np.nan)
        longs = shorts = 0
        if len(index) >= min_assets:
            z = session_z(score)
            before = previous_smooth[index]
            smooth = np.where(np.isnan(before), z, smoothing * z + (1 - smoothing) * before)
            smooth_now[index] = smooth
            position = (rankdata(smooth) - 0.5) / len(smooth)
            held = previous_weights[index]
            long_side = (position >= 1 - entry) | ((held > 0) & (position >= 1 - exit_tail))
            short_side = (position < entry) | ((held < 0) & (position < exit_tail))
            if long_side.any() and short_side.any():
                weights[index[long_side]] = 1.0 / long_side.sum()
                weights[index[short_side]] = -1.0 / short_side.sum()
                longs, shorts = int(long_side.sum()), int(short_side.sum())
        gross = float(weights[index] @ returns) if len(index) else 0.0
        turnover = float(np.abs(weights - previous_weights).sum())
        result.append((key, gross, turnover, longs, shorts))
        previous_smooth, previous_weights = smooth_now, weights
    return result


def net_series(book, cost_bps):
    """Neto por sesión a un coste por lado: bruto − c·rotación."""
    return np.array([gross - cost_bps * 1e-4 * turnover for _, gross, turnover, _, _ in book])


def choose(sessions, assets, settings, min_assets):
    """(λ, salida) de mayor neto medio en calibración al coste declarado; a igualdad, el más
    sencillo (λ mayor y, después, salida menor)."""
    cost = settings["choice"]["cost_bps"]
    entry = settings["band"]["entry_tail"]
    table = []
    for smoothing in settings["smoothing"]["lambdas"]:
        for exit_tail in settings["band"]["exit_tails"]:
            book = run_book(
                sessions,
                assets,
                smoothing=smoothing,
                entry=entry,
                exit_tail=exit_tail,
                min_assets=min_assets,
            )
            table.append(
                dict(
                    smoothing=smoothing,
                    exit_tail=exit_tail,
                    mean_net_bps=1e4 * float(net_series(book, cost).mean()) if book else None,
                    turnover=float(np.mean([b[2] for b in book])) if book else None,
                )
            )
    ranked = sorted(
        table,
        key=lambda r: (
            -(r["mean_net_bps"] if r["mean_net_bps"] is not None else -math.inf),
            -r["smoothing"],
            r["exit_tail"],
        ),
    )
    return dict(smoothing=ranked[0]["smoothing"], exit_tail=ranked[0]["exit_tail"]), table


def drawdown(values):
    """Caída máxima de la suma acumulada."""
    path = np.cumsum(values)
    return float(np.max(np.maximum.accumulate(np.r_[0.0, path]) - np.r_[0.0, path]))


def book_measures(tool, book, settings):
    """Medidas de un libro en evaluación: por coste y por horizonte."""
    keys = [b[0] for b in book]
    gross = np.array([b[1] for b in book])
    turnover = np.array([b[2] for b in book])
    annual = math.sqrt(settings["annualization"])
    result = dict(
        sessions=len(book),
        gross_bps=1e4 * float(gross.mean()) if len(book) else None,
        turnover=float(turnover.mean()) if len(book) else None,
        breakeven_bps=1e4 * float(gross.mean() / turnover.mean())
        if len(book) and turnover.mean() > 0
        else None,
        names=dict(
            long=float(np.mean([b[3] for b in book])) if book else None,
            short=float(np.mean([b[4] for b in book])) if book else None,
        ),
        costs={},
    )
    for cost in settings["costs_bps_per_side"]:
        net = net_series(book, cost)
        stats = tool.summary(dict(zip(keys, net.tolist(), strict=True)))
        spread = float(net.std(ddof=1)) if len(net) > 1 else 0.0
        horizons = {}
        for h in settings["horizons"]:
            count = len(net) // h
            windows = net[: count * h].reshape(count, h).sum(axis=1) if count else np.array([])
            entry = dict(
                windows=int(count),
                mean_bps=1e4 * float(windows.mean()) if count else None,
                positive=rel.wilson(int((windows > 0).sum()), int(count)),
            )
            if h == 1:
                entry["positive_block"] = rel.block_interval(
                    net > 0, keys, settings["block_interval"]
                )
            horizons[str(h)] = entry
        result["costs"][str(cost)] = dict(
            mean_bps=1e4 * float(net.mean()) if len(net) else None,
            t_nw=stats["t_nw"],
            sharpe=float(net.mean() / spread * annual) if spread > 0 else None,
            max_drawdown_bps=1e4 * drawdown(net) if len(net) else None,
            horizons=horizons,
        )
    return result


def contrast(tool, left, right, cost):
    """Neto de `left` − neto de `right` por sesión a un coste: media, t de Newey-West e
    intervalo al 95 %."""
    a = dict(zip([b[0] for b in left], net_series(left, cost).tolist(), strict=True))
    b = dict(zip([b[0] for b in right], net_series(right, cost).tolist(), strict=True))
    common = sorted(set(a) & set(b))
    stats = tool.summary({k: a[k] - b[k] for k in common})
    t = stats["t_nw"]
    se = (
        abs(stats["mean"] / t)
        if t not in (None, 0) and stats["mean"] is not None and math.isfinite(t)
        else None
    )
    return dict(
        sessions=len(common),
        mean_bps=None if stats["mean"] is None else 1e4 * stats["mean"],
        t_nw=t,
        interval_bps=None
        if se is None
        else [1e4 * (stats["mean"] - rel.Z95 * se), 1e4 * (stats["mean"] + rel.Z95 * se)],
    )


def verdict(scopes, rule):
    """La regla declarada, leída mecánicamente en el alcance declarado."""
    item = scopes[rule["scope"]]
    versus = item["layer_vs_control"][str(rule["beats_control_cost_bps"])]
    own = item["layer"]["costs"][str(rule["positive_cost_bps"])]
    reasons = []
    if (
        versus["mean_bps"] is None
        or versus["mean_bps"] <= 0
        or versus["t_nw"] is None
        or versus["t_nw"] < rule["beats_control_min_t"]
    ):
        reasons.append(
            f"a {rule['beats_control_cost_bps']} pb la capa no supera al control con t ≥ {rule['beats_control_min_t']}"
        )
    if (
        own["mean_bps"] is None
        or own["mean_bps"] <= 0
        or own["t_nw"] is None
        or own["t_nw"] < rule["positive_min_t"]
    ):
        reasons.append(
            f"a {rule['positive_cost_bps']} pb su neto no es > 0 con t ≥ {rule['positive_min_t']}"
        )
    if (
        item["chosen"]["smoothing"] == 1.0
        and item["chosen"]["exit_tail"] == item["control_parameters"]["exit_tail"]
    ):
        reasons.append("en calibración se eligió el propio control")
    return dict(scope=rule["scope"], enters=not reasons, reasons=reasons)


# ---------------------------------------------------------------------------------------------
# La orden


def _sessions(view, assets_of, keep, score):
    """(clave, índices de activo, puntuación, rendimiento) por sesión, en el orden del tiempo."""
    result = []
    for key, (start, end) in sorted(
        zip(view["keys"], view["bounds"], strict=True), key=lambda x: (x[0][1], x[0][0])
    ):
        rows = start + np.flatnonzero(keep[start:end])
        result.append(
            (
                key,
                np.array([assets_of[view["asset_id"][r]] for r in rows], dtype=np.int64),
                score[rows],
                view["target"][rows],
            )
        )
    return result


def measure(
    declaration, state, reference, organs, origin, output, *, wait_hours, poll, log, markets=None
):
    """P de un origen, publicada en `<output>/<origen>` (o `<origen>-<mercados>`)."""
    declaration, digest = checked_declaration(declaration)
    sources = og.folders(declaration, state, reference, organs, origin)
    manifest = Path(state) / "cohorts" / str(origin) / "manifest.json"
    missing = og.wait_for_inputs(
        declaration, sources, manifest, hours=wait_hours, poll=poll, say=log
    )
    if missing:
        raise og.MissingInputs(missing)
    fingerprint = og._fingerprint(sources, manifest)
    target = Path(output) / og.target_name(origin, markets)
    if target.exists():
        return og._already(target, digest, fingerprint, log)
    started = time.monotonic()

    def say(text):
        log(f"{time.monotonic() - started:6.0f} s {text}")

    records = {
        role: json.loads((folder / "run.json").read_text(encoding="utf-8"))
        for role, folder in sources.items()
    }
    og._checked_sources(declaration, origin, sources, records, fingerprint)
    tool = og._screening()
    parent = {
        s: pq.read_table(
            sources["parent"] / f"{s}-predictions.parquet", columns=og.COLUMNS
        ).sort_by(og.ORDER)
        for s in SPLITS
    }
    tables = dict(
        tree={s: og._table(sources["tree"], s) for s in SPLITS},
        organ={s: og._table(sources["organ"], s) for s in AGGREGATED},
    )
    present = og.markets_of(parent)
    measured = og.measured_markets(present, markets)
    if measured != present:
        parent = {s: og.only(table, measured) for s, table in parent.items()}
        tables = {
            role: {s: og.only(table, measured) for s, table in splits.items()}
            for role, splits in tables.items()
        }
    say(f"mercados medidos: {measured} (el estado tiene {present})")
    for role, splits in tables.items():
        for split, table in splits.items():
            for name in og.KEYS:
                _require(
                    table.column(name).equals(parent[split].column(name)),
                    f"{sources[role]} no tiene las filas del padre en {split} (columna {name})",
                )
    view = records["parent"]["request"]["view_sha256"]
    alpha, liquidity, _, matures = ac.known_rows(state, origin, parent, view)
    universe = declaration["universe"]
    cap = rel.alpha_cap(
        alpha,
        dict(
            alpha_cap_split=universe["alpha_cap_split"],
            alpha_cap_quantile=universe["alpha_cap_quantile"],
        ),
    )
    say(f"alfa conocido, liquidez y madurez; tope de |α| {cap:.6f}")
    own = {s: parent[s].column("prediction").to_numpy().astype(np.float64) for s in SPLITS}
    predictions = dict(
        parent=own,
        tree={
            s: tables["tree"][s].column("prediction").to_numpy().astype(np.float64) for s in SPLITS
        },
        signal={
            s: tables["organ"][s].column("prediction").to_numpy().astype(np.float64) - own[s]
            for s in AGGREGATED
        },
    )
    names = dict(parent=declaration["parent"], tree=declaration["tree"], signal=og.SIGNAL)
    min_assets = declaration["min_assets"]
    Path(output).mkdir(parents=True, exist_ok=True)
    work = Path(output) / f".{origin}.{os.getpid()}.tmp"
    try:
        work.mkdir(parents=True)
        inputs = {
            p: og._write_component(
                work / "entradas" / names[p], parent, predictions[p], alpha, matures
            )
            for p in names
        }
        results = {
            system: oa.run_aggregation(
                {names[p]: inputs[p] for p in members},
                splits=tuple(declaration["splits"]),
                selection_split=declaration["selection_split"],
                min_assets=min_assets,
                align=declaration["align"],
                operable=True,
                alpha_col="alpha",
                maturity_col="matures_at",
                placebo_seed=declaration["aggregator_placebo_seed"],
                operable_scales=tuple(declaration["operable_scales"]),
            )
            for system, members in rel.SYSTEMS
        }
        views = rel._views("operable", results, names, parent, liquidity, alpha, cap)
        say("S2 operable agregado")
        assets_of = {}
        for split in AGGREGATED:
            aligned = results["s2"]["aligned"][split]
            for item in (views[("s2", split)],):
                item["asset_id"] = list(aligned["asset_id"])
            for asset in aligned["asset_id"]:
                assets_of.setdefault(asset, len(assets_of))
        count = len(assets_of)
        book = declaration["book"]
        entry = book["tail"]
        scopes, rows = {}, []
        for scope in universe["scopes"]:
            data = {}
            for split in AGGREGATED:
                item = views[("s2", split)]
                keep = ~item["outlier"]
                if scope == "liquid":
                    keep &= rel.liquid_rows(item["liquidity"], item["bounds"])
                data[split] = dict(
                    s2=_sessions(item, assets_of, keep, item["score"]),
                    alpha=_sessions(item, assets_of, keep, item["alpha"]),
                )
            chosen, table = choose(data["calibration"]["s2"], count, declaration, min_assets)
            books = dict(
                layer=run_book(
                    data["evaluation"]["s2"],
                    count,
                    smoothing=chosen["smoothing"],
                    entry=entry,
                    exit_tail=chosen["exit_tail"],
                    min_assets=min_assets,
                ),
                control=run_book(
                    data["evaluation"]["s2"],
                    count,
                    smoothing=1.0,
                    entry=entry,
                    exit_tail=entry,
                    min_assets=min_assets,
                ),
                alpha=run_book(
                    data["evaluation"]["alpha"],
                    count,
                    smoothing=1.0,
                    entry=entry,
                    exit_tail=entry,
                    min_assets=min_assets,
                ),
            )
            scopes[scope] = dict(
                chosen=chosen,
                calibration_grid=table,
                control_parameters=dict(smoothing=1.0, exit_tail=entry),
                **{name: book_measures(tool, b, declaration) for name, b in books.items()},
                layer_vs_control={
                    str(c): contrast(tool, books["layer"], books["control"], c)
                    for c in declaration["costs_bps_per_side"]
                },
                layer_vs_alpha={
                    str(c): contrast(tool, books["layer"], books["alpha"], c)
                    for c in declaration["costs_bps_per_side"]
                },
            )
            for name, b in books.items():
                rows += [
                    dict(
                        scope=scope,
                        book=name,
                        market=str(k[0]),
                        prediction_at=int(k[1]),
                        gross=g,
                        turnover=t,
                        longs=lo,
                        shorts=sh,
                    )
                    for k, g, t, lo, sh in b
                ]
            say(f"{scope}: elegido λ = {chosen['smoothing']}, salida {chosen['exit_tail']}")
        shutil.rmtree(work / "entradas")
        code = og._code_identity()
        code["files"]["portfolio_layer"] = pilots._sha256(Path(__file__))
        code["files"]["reliability"] = pilots._sha256(Path(rel.__file__))
        summary = og._plain(
            dict(
                schema_version=1,
                kind=KIND,
                status="completed",
                origin=origin,
                declaration_sha256=digest,
                code=code,
                inputs=fingerprint,
                markets=measured,
                alpha_cap=cap,
                operable_factors=results["s2"]["operable"]["scales"],
                scopes=scopes,
                verdict=verdict(scopes, declaration["rule"]),
                seconds=round(time.monotonic() - started),
            )
        )
        pq.write_table(ac._series_table(rows), work / "sesiones.parquet")
        (work / "resumen.md").write_text(markdown(summary), encoding="utf-8")
        (work / "resumen.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        os.rename(work, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    say(f"publicado en {target}")
    return summary


def _f(value, digits=2):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",").replace("-", "−")


def _frac(entry):
    if not entry:
        return "—"
    return f"{_f(entry['rate'], 3)} [{_f(entry['low'], 3)}; {_f(entry['high'], 3)}]"


def markdown(summary):
    """Resumen legible de P de un origen."""
    result = summary["verdict"]
    lines = [
        f"# P, la capa de cartera, en el origen {summary['origin']}",
        "",
        f"Declaración {summary['declaration_sha256'][:12]}; mercados {', '.join(summary['markets'])}; "
        f"evaluación; puntuación operable de S2 sin |α| sobre {_f(summary['alpha_cap'], 5)}.",
        "",
        f"**Regla ({result['scope']}): "
        + ("entra." if result["enters"] else "no entra: " + "; ".join(result["reasons"]) + ".")
        + "**",
        "",
    ]
    for scope, item in summary["scopes"].items():
        chosen = item["chosen"]
        lines += [
            f"## {'Tercio líquido' if scope == 'liquid' else 'Todos los nombres'}",
            "",
            f"Elegido en calibración: λ = {_f(chosen['smoothing'])}, salida {_f(chosen['exit_tail'])}.",
            "",
            "| Libro | Bruto (pb) | Rotación | Equilibrio (pb) | Coste | Neto (pb) | t | Sharpe | Caída máxima (pb) | Sesiones > 0 | Ventanas de 5 > 0 | Ventanas de 21 > 0 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name in ("layer", "control", "alpha"):
            book = item[name]
            for cost, part in book["costs"].items():
                h = part["horizons"]
                lines.append(
                    f"| {dict(layer='capa', control='control', alpha='alfa solo')[name]} | {_f(book['gross_bps'])} | "
                    f"{_f(book['turnover'], 3)} | {_f(book['breakeven_bps'])} | {cost} | {_f(part['mean_bps'])} | "
                    f"{_f(part['t_nw'])} | {_f(part['sharpe'])} | {_f(part['max_drawdown_bps'])} | "
                    f"{_frac(h['1']['positive'])} | {_frac(h['5']['positive'])} | {_frac(h['21']['positive'])} |"
                )
        lines += [
            "",
            "| Capa − control | Media (pb) | t | Intervalo al 95 % (pb) |",
            "|---|---|---|---|",
        ]
        for cost, part in item["layer_vs_control"].items():
            interval = part["interval_bps"]
            lines.append(
                f"| {cost} pb | {_f(part['mean_bps'])} | {_f(part['t_nw'])} | "
                + ("—" if not interval else f"[{_f(interval[0])}; {_f(interval[1])}]")
                + " |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def parser():
    result = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = result.add_subparsers(dest="command", required=True)
    run = commands.add_parser("medir", help="Calcular P de un origen")
    run.add_argument("--declaration", type=Path, required=True)
    run.add_argument("--state", type=Path, required=True, help="Estado de la campaña")
    run.add_argument("--reference", type=Path, required=True, help="Raíz de los árboles")
    run.add_argument("--organs", type=Path, required=True, help="Raíz de los órganos")
    run.add_argument("--origin", type=int, required=True)
    run.add_argument("--output", type=Path, required=True, help="Raíz propia de P")
    run.add_argument("--wait-hours", type=float, default=0.0)
    run.add_argument("--poll-seconds", type=float, default=300.0)
    run.add_argument(
        "--market",
        action="append",
        dest="markets",
        metavar="MERCADO",
        help="Medir solo este mercado (repetible); obligatorio en un estado con varios",
    )
    return result


def main(argv=None):
    args = parser().parse_args(argv)

    def log(line):
        print(f"{time.strftime('%H:%M:%SZ', time.gmtime())} {line}", flush=True)

    try:
        declaration = json.loads(args.declaration.read_text(encoding="utf-8"))
        measure(
            declaration,
            args.state,
            args.reference,
            args.organs,
            args.origin,
            args.output,
            wait_hours=args.wait_hours,
            poll=args.poll_seconds,
            log=log,
            markets=args.markets,
        )
    except og.MissingInputs as missing:
        print(
            f"Faltan entradas de P para el origen {args.origin} tras esperar "
            f"{args.wait_hours:g} h; no se escribe nada:",
            file=sys.stderr,
        )
        for item in missing.items:
            print(f"- {item}", file=sys.stderr)
        return og.MISSING_EXIT
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
