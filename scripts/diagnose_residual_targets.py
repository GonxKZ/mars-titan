"""Diagnóstico de la etiqueta residual (MT-016) y sus figuras.

`run` lee unos objetivos confirmados y su edición y escribe el informe JSON. No ajusta ningún
modelo ni abre la prueba final. `figures` dibuja las figuras desde ese informe, comprobando su
huella, y deja junto a ellas un manifiesto con la huella del informe y la de cada figura.

    uv run python scripts/diagnose_residual_targets.py run \\
        --targets <objetivos>/manifest.json --edition <edición>/manifest.json \\
        --declaration configs/targets/residual-diagnostics-v1.json \\
        --output reports/targets/residual-diagnostics-v3-20261010.json --workers 4
    uv run python scripts/diagnose_residual_targets.py figures \\
        --report reports/targets/residual-diagnostics-v3-20261010.json \\
        --output reports/targets/figures/v3
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mars_titan.data import residual_diagnostics as rd  # noqa: E402
from mars_titan.data.storage import atomic_json, sha256  # noqa: E402

# Grupos de categorías de la figura de cobertura, en el orden de apilado.
GROUPS = {
    "Entrenamiento": ("train",),
    "Validación": ("validation",),
    "Historia corta del activo": ("insufficient_stock_history",),
    "Factor ausente en la ventana": ("missing_factor_history",),
    "Sesión siguiente ausente o inválida": (
        "next_stock_session_absent",
        "next_stock_price_invalid",
    ),
    "Factor ausente al día siguiente": ("next_factor_return_missing",),
    "Otras exclusiones": (
        "zero_market_variance",
        "target_after_cutoff",
        "target_crosses_partition_boundary",
        "outside_label_calendar",
    ),
}


def run(args):
    declaration = rd.load_declaration(args.declaration)
    for source in (args.targets.parent, args.edition.parent):
        if args.output.resolve().is_relative_to(source.resolve()):
            raise ValueError("El informe no se escribe dentro de los objetivos ni de la edición")
    started = time.perf_counter()

    def progress(done, total):
        if done % 250 == 0 or done == total:
            print(f"{done}/{total} activos", file=sys.stderr, flush=True)

    report = rd.diagnose(
        args.targets,
        args.edition,
        declaration,
        workers=args.workers,
        patterns=not args.without_patterns,
        progress=progress,
    )
    report["seconds"] = time.perf_counter() - started
    report["peak_rss_bytes"] = {
        "parent": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "largest_worker": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * 1024,
    }
    atomic_json(args.output, report)
    print(json.dumps(dict(output=str(args.output), sha256=sha256(args.output))))


def _save(figure, path):
    figure.savefig(path, metadata={"Date": None, "Creator": "Matplotlib"})
    plt.close(figure)
    return sha256(path)


def _coverage(report, axes):
    for axis, (market, years) in zip(
        axes, report["coverage"]["samples"]["by_market_and_year"].items(), strict=True
    ):
        labels = sorted(years, key=int)
        bottom = [0] * len(labels)
        for name, categories in GROUPS.items():
            values = [sum(years[y].get(c, 0) for c in categories) for y in labels]
            if any(values):
                axis.bar(labels, values, bottom=bottom, label=name, width=0.8)
                bottom = [a + b for a, b in zip(bottom, values, strict=True)]
        axis.set_title(f"{market}: muestras por año y categoría")
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[-1].legend(fontsize=7, loc="upper left")


def _monthly(report, axes):
    for axis, (market, months) in zip(axes, report["monthly_beta"].items(), strict=True):
        x = list(range(len(months)))
        for key, style in (("p10", ":"), ("p50", "-"), ("p90", ":")):
            axis.plot(x, [m[key] for m in months], style, label=key, linewidth=1)
        ticks = [i for i, m in enumerate(months) if m["month"].endswith("-01")]
        axis.set_xticks(ticks, [months[i]["month"][:4] for i in ticks], rotation=90, fontsize=7)
        axis.axhline(1.0, color="grey", linewidth=0.5)
        axis.set_title(f"{market}: beta transversal por mes")
    axes[-1].legend(fontsize=7)


def _yearly(report, axes, keys, title):
    for axis, (market, years) in zip(axes, report["by_year"].items(), strict=True):
        labels = sorted(years, key=int)
        for label, read in keys:
            axis.plot(labels, [read(years[y]) for y in labels], marker=".", label=label)
        axis.axhline(0.0, color="grey", linewidth=0.5)
        axis.set_title(f"{market}: {title}")
        axis.tick_params(axis="x", labelrotation=90, labelsize=7)
    axes[-1].legend(fontsize=7)


def figures(args):
    report = json.loads(args.report.read_text())
    if report.get("kind") != rd.KIND or report.get("final_test_opened") is not False:
        raise ValueError("El archivo no es un informe del diagnóstico residual")
    plt.rcParams["svg.hashsalt"] = "mars-titan-residual-diagnostics"
    args.output.mkdir(parents=True, exist_ok=True)
    drawn = {}
    plans = {
        "coverage-by-year": lambda axes: _coverage(report, axes),
        "beta-by-month": lambda axes: _monthly(report, axes),
        "dispersion-by-year": lambda axes: _yearly(
            report,
            axes,
            [("bruto", lambda y: y["std_raw"]), ("residual", lambda y: y["std_residual"])],
            "desviación típica diaria",
        ),
        "exposure-by-year": lambda axes: _yearly(
            report,
            axes,
            [
                ("residual", lambda y: y["residual_on_factor"]["correlation"]),
                ("bruto", lambda y: y["raw_on_factor"]["correlation"]),
            ],
            "correlación con el factor del día siguiente",
        ),
    }
    for name, draw in plans.items():
        figure, axes = plt.subplots(1, len(report["by_year"]), figsize=(12, 4.2))
        draw(list(axes) if hasattr(axes, "__len__") else [axes])
        figure.tight_layout()
        drawn[f"{name}.svg"] = _save(figure, args.output / f"{name}.svg")
    atomic_json(
        args.output / "figures.json",
        dict(
            report=str(args.report),
            report_sha256=sha256(args.report),
            matplotlib=matplotlib.__version__,
            figures=drawn,
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    diagnose = commands.add_parser("run")
    diagnose.add_argument("--targets", type=Path, required=True)
    diagnose.add_argument("--edition", type=Path, required=True)
    diagnose.add_argument("--declaration", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    diagnose.add_argument("--workers", type=int, default=1, choices=range(1, 9))
    diagnose.add_argument(
        "--without-patterns",
        action="store_true",
        help="No leer la presencia de las muestras para el desglose por patrón de máscaras",
    )
    draw = commands.add_parser("figures")
    draw.add_argument("--report", type=Path, required=True)
    draw.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    (run if args.command == "run" else figures)(args)


if __name__ == "__main__":
    main()
