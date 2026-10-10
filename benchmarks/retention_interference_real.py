"""Comprobar con datos reales y medir el análisis de retención e interferencia.

Usa los precios preparados de la edición desde 2000 y los protocolos walk-forward v2. No hay
modelos, predicciones aprendidas ni pasos de optimizador:

1. Construye el calendario de regímenes de US y CN y mide su tiempo y su memoria.
2. Recorta la historia en varias fechas y exige que las rutas anteriores no cambien.
3. Recalcula una muestra de sesiones con las ventanas de la codificación
   (`price_windows.window_rows`) y la regla de `memory.regimes`.
4. Cuenta las clases de las sesiones de evaluación de cada mercado con la declaración de la
   comparación, con las rutas reales y con el placebo.
5. Sobre la rejilla real de sesiones de evaluación, inyecta un beneficio conocido por clase y
   exige que el informe lo recupere, y que un par idéntico dé cero. Mide el informe completo
   con los ocho pares declarados y las réplicas de la comparación.

Los beneficios inyectados no son resultados de ningún modelo. El recibo no contiene errores
de predicción ni estadísticos de ningún brazo.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from mars_titan.data.price_windows import CONTEXT_SESSIONS, window_rows
from mars_titan.data.storage import atomic_json
from mars_titan.evaluation import regime_calendar as rc
from mars_titan.evaluation import retention_interference as ri
from mars_titan.evaluation.splits import build_folds
from mars_titan.memory.regimes import REGIME_RULE, UNCLASSIFIED, RegimeRule

ROOT = Path(__file__).resolve().parents[1]
COMPARISON = ROOT / "configs/evaluation/historical-masked-2000-comparison.json"
PROTOCOLS = {
    "US": ROOT / "configs/evaluation/historical-masked-us-walk-forward-v2.json",
    "CN": ROOT / "configs/evaluation/historical-masked-cn-walk-forward-v2.json",
}
CUTS = {"US": ("2008-09-15", "2020-03-16"), "CN": ("2008-09-16", "2015-06-12")}
SAMPLE = 40
# Beneficio inyectado por clase: nuevo, revisita corta, revisita larga y asentado.
INJECTED = dict(novel=0.1, short=0.3, long=0.2, continuing=0.0)


def _rss_mib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _check_prefixes(prepared, absent_records, calendar, rule):
    """Rutas iguales hasta cada corte con la historia recortada en ese corte."""
    result = {}
    for market, cuts in CUTS.items():
        sessions, at, closes, absent = rc._read_market(
            prepared / "prepared", market, absent_records[market]
        )
        full = calendar["markets"][market]
        for cut in cuts:
            keep = sum(1 for day in sessions if day <= cut)
            started = time.perf_counter()
            prefix = rc.market_routes(
                market, sessions[:keep], at[:keep], closes[:, :keep], absent[:keep], rule
            )
            rows = len(prefix["sessions"])
            same = all(prefix[name] == full[name][:rows] for name in rc._FIELDS)
            if not same:
                raise SystemExit(f"Las rutas de {market} cambian al recortar en {cut}")
            result[f"{market}/{cut}"] = dict(
                sessions=rows, identical=same, seconds=time.perf_counter() - started
            )
    return result


def _check_sample(prepared, absent_records, calendar, rule, seed):
    """Muestra de sesiones recalculada con las ventanas de la codificación."""
    rng = np.random.default_rng(seed)
    result = {}
    for market in CUTS:
        sessions, at, closes, absent = rc._read_market(
            prepared / "prepared", market, absent_records[market]
        )
        record = calendar["markets"][market]
        index = {day: i for i, day in enumerate(record["sessions"])}
        positions_absent = np.flatnonzero(absent)
        decisions = np.flatnonzero(~absent)
        decisions = decisions[decisions >= CONTEXT_SESSIONS - 1]
        chosen = np.sort(rng.choice(decisions, size=SAMPLE, replace=False))
        matched = 0
        for end in chosen.tolist():
            windows = []
            for row in closes:
                positions = np.flatnonzero(np.isfinite(row))
                if end not in positions:
                    continue
                try:
                    rows = window_rows(
                        positions,
                        np.array([np.searchsorted(positions, end)]),
                        CONTEXT_SESSIONS,
                        positions_absent,
                    )[0]
                except ValueError:
                    continue
                present = rows >= 0
                logs = np.where(present, np.log(row[positions[np.maximum(rows, 0)]]), 0.0)
                window = np.zeros((CONTEXT_SESSIONS, 6))
                window[present, 3] = logs[present] - logs[np.argmax(present)]
                window[:, 5] = present
                windows.append(window)
            if not windows:
                raise SystemExit(f"{market} {sessions[end]}: ninguna ventana válida en la muestra")
            state = rule.state(market, np.stack(windows), int(at[end]))
            i = index[sessions[end]]
            expected = (state.route, state.assets, state.returns)
            found = tuple(record[name][i] for name in ("route", "assets", "returns"))
            if expected != found:
                raise SystemExit(f"{market} {sessions[end]}: {found} frente a {expected}")
            matched += 1
        result[market] = dict(sampled=SAMPLE, matched=matched, first=str(sessions[chosen[0]]))
    return result


def _windows(market):
    protocol = json.loads(PROTOCOLS[market].read_text())
    return [tuple(fold["evaluation"]) for fold in build_folds(protocol)]


def _grid(calendar, market, windows):
    """Instantes de las sesiones de evaluación y su día UTC renumerado."""
    at, _ = calendar[market]
    first = int(np.datetime64(windows[0][0], "us").astype(np.int64))
    last = int(np.datetime64(windows[-1][1], "us").astype(np.int64))
    times = at[(at >= first) & (at < last)]
    days = np.floor_divide(times, 86_400_000_000)
    return times, np.unique(days, return_inverse=True)[1]


def _injected(kind, absence, long_absence):
    revisit = kind == ri.CLASSES.index("revisit_entry")
    return np.select(
        [
            kind == ri.CLASSES.index("novel_entry"),
            revisit & (absence < long_absence),
            revisit,
        ],
        [INJECTED["novel"], INJECTED["short"], INJECTED["long"]],
        INJECTED["continuing"],
    )


def _check_report(calendar, section, options, repeats):
    """Beneficios inyectados en la rejilla real: recuperación exacta y coste del informe."""
    result = {}
    for market in CUTS:
        windows = _windows(market)
        times, period = _grid(calendar, market, windows)
        control = 1.0 + 0.1 * np.sin(np.arange(len(times)) / 7.0)
        series = {}
        expected = {}
        for name, pair in section["pairs"].items():
            kind, absence, _ = ri.session_classes(
                times, calendar[market], windows, section, pair["history"]
            )
            benefit = _injected(kind, absence, section["long_absence_sessions"])
            series[pair["memory"] + "@" + name] = control - benefit
            expected[name] = (kind, absence, benefit)
        twin = dict(memory="twin", control="control", history=ri.STARTS[0], separates="identity")

        # Cada par usa su propia memoria inyectada y el mismo control.
        pairs = {
            name: dict(pair, memory=f"{pair['memory']}@{name}", control="control")
            for name, pair in section["pairs"].items()
        }
        pairs["identity"] = twin

        def lookup(arm, _market, times=times, period=period, control=control, series=series):
            values = control if arm in ("control", "twin") else series[arm]
            return values, np.ones(len(times), bool), times, period

        declared = dict(section, pairs=pairs)
        samples, report = [], None
        for _ in range(repeats):
            started = time.perf_counter()
            report = ri.report(declared, options, windows, lookup, calendar, [market])
            samples.append(time.perf_counter() - started)
        records = report["markets"][market]["pairs"]
        checks = {}
        for name, (kind, absence, benefit) in expected.items():
            estimates = records[name]["estimates"]
            novel = benefit[kind == 0].mean() if np.any(kind == 0) else np.nan
            revisit = benefit[kind == 1].mean()
            wanted = revisit - novel
            got = estimates["retention"]
            exact = bool((np.isnan(wanted) and got is None) or abs(got - wanted) < 1e-12)
            if not exact:
                raise SystemExit(f"{market} {name}: retención {got} frente a {wanted}")
            long = benefit[(kind == 1) & (absence >= section["long_absence_sessions"])].mean()
            short = benefit[(kind == 1) & (absence < section["long_absence_sessions"])].mean()
            if abs(estimates["interference"] - (long - short)) >= 1e-12:
                raise SystemExit(f"{market} {name}: la interferencia no se recupera")
            checks[name] = dict(
                history=section["pairs"][name]["history"],
                sessions=records[name]["sessions"],
                retention_recovered=exact,
                interference_recovered=True,
                decisions={
                    key: value["status"] for key, value in records[name]["decisions"].items()
                },
            )
        identity = records["identity"]["estimates"]
        if any(value not in (0.0, None) for value in identity.values()):
            raise SystemExit(f"{market}: el par idéntico no da cero")
        result[market] = dict(
            windows=len(windows),
            sessions=len(times),
            days=int(period.max()) + 1,
            pairs=checks,
            identity_estimates=identity,
            report_seconds=dict(p50=statistics.median(samples), min=min(samples), repeats=repeats),
        )
    return result


def _class_counts(calendar, section):
    counts = {}
    for market in CUTS:
        windows = _windows(market)
        times, _ = _grid(calendar, market, windows)
        for start in ri.STARTS:
            for placebo in (False, True):
                kind, absence, _ = ri.session_classes(
                    times, calendar[market], windows, section, start, placebo=placebo
                )
                revisit = kind == ri.CLASSES.index("revisit_entry")
                key = f"{market}/{start}/{'placebo' if placebo else 'real'}"
                counts[key] = {
                    name: int(np.sum(kind == i)) for i, name in enumerate(ri.CLASSES)
                } | dict(
                    unclassified=int(np.sum(kind < 0)),
                    revisit_long=int(
                        np.sum(revisit & (absence >= section["long_absence_sessions"]))
                    ),
                    revisit_short=int(
                        np.sum(revisit & (absence < section["long_absence_sessions"]))
                    ),
                )
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--calendar", type=Path, required=True, help="Calendario nuevo")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    comparison = json.loads(COMPARISON.read_text())
    section = comparison["retention_interference"]
    options = {
        key: comparison["comparison"][key]
        for key in ("block_length", "replicates", "seed", "confidence")
    }
    rule = RegimeRule(REGIME_RULE)
    load = os.getloadavg()
    started = time.perf_counter()
    calendar = rc.build(args.prepared, rule=rule)
    build_seconds = time.perf_counter() - started
    build_rss = _rss_mib()
    digest = rc.write(calendar, args.calendar, sources=(args.prepared,))
    resolved, _, _ = rc.read(args.calendar)
    absent_records = json.loads((args.prepared / rc.ABSENT_FILE).read_text())
    prefixes = _check_prefixes(args.prepared, absent_records, calendar, rule)
    sample = _check_sample(args.prepared, absent_records, calendar, rule, seed=20261010)
    counts = _class_counts(resolved, section)
    reports = _check_report(resolved, section, options, args.repeats)
    receipt = dict(
        schema_version=1,
        kind="retention_interference_real_data_check",
        issue=54,
        measured_at=datetime.now(UTC).isoformat(timespec="seconds"),
        commit=subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip(),
        sources_sha256={
            path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
            for path in (
                "src/mars_titan/evaluation/regime_calendar.py",
                "src/mars_titan/evaluation/retention_interference.py",
                "src/mars_titan/memory/regimes.py",
                "configs/evaluation/historical-masked-2000-comparison.json",
                "benchmarks/retention_interference_real.py",
            )
        },
        data=dict(
            prepared=str(args.prepared),
            prepared_manifest_sha256=calendar["prepared"]["manifest_sha256"],
            market_absent_sessions_sha256=calendar["prepared"]["market_absent_sessions_sha256"],
            calendar_sha256=digest,
            note="precios reales preparados hasta 2023. Ningún modelo, predicción ni objetivo",
        ),
        calendar=dict(
            build_seconds=build_seconds,
            process_peak_rss_mib_after_build=build_rss,
            summary=rc.summary(calendar),
        ),
        prefix_checks=prefixes,
        sample_checks=sample,
        class_counts=counts,
        injected=dict(values=INJECTED, options=options, markets=reports),
        environment=dict(
            cpu=next(
                (
                    line.split(":", 1)[1].strip()
                    for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")
                ),
                platform.processor(),
            ),
            numpy=np.__version__,
            python=platform.python_version(),
            omp_threads=os.environ.get("OMP_NUM_THREADS"),
            load_average_at_start=load,
            process_peak_rss_mib=_rss_mib(),
        ),
        not_measured=[
            "GPU, porque el calendario y el análisis trabajan en CPU con NumPy",
            "energía, sin instrumento",
            "el informe completo de la comparación, que necesita predicciones de la campaña",
        ],
        unclassified_rule=UNCLASSIFIED,
    )
    atomic_json(args.output, receipt)
    print(
        json.dumps(dict(calendar=receipt["calendar"]["build_seconds"], reports=reports), indent=1)
    )


if __name__ == "__main__":
    main()
