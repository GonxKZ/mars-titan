"""Exportar tablas agregadas verificadas y figuras de las campañas comparativas."""

import argparse
import csv
import math
from collections import defaultdict
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, outside_source, sha256


def rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def predictions(directory, folds):
    records = rows(directory / "predictive-intervals.csv")
    starts = {row["id"]: date.fromisoformat(row["windows"]["evaluation"][0]) for row in folds}
    months = (
        "ene.",
        "feb.",
        "mar.",
        "abr.",
        "may.",
        "jun.",
        "jul.",
        "ago.",
        "sep.",
        "oct.",
        "nov.",
        "dic.",
    )
    methods = (
        ("reference", "Referencia continua", "#245f9e", "o"),
        ("neural_mae", "Continuación MAE (rejilla)", "#b26322", "s"),
        ("neural_mse", "Continuación MSE (rejilla)", "#706223", "^"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for ax, family in zip(axes.flat, ("rnn", "lstm", "gru", "dlinear"), strict=True):
        for offset, (method, label, color, marker) in zip((-0.2, 0, 0.2), methods, strict=True):
            chosen = sorted(
                (
                    r
                    for r in records
                    if r["partition"] == "evaluation"
                    and r["family"] == family
                    and r["method"] == method
                    and r["reference"] == "zero"
                    and r["block_length"] == "5"
                ),
                key=lambda r: r["fold"],
            )
            if len(chosen) != 4:
                raise ValueError("La figura necesita las cuatro ventanas revisadas")
            estimate = np.array([float(r["estimate"]) for r in chosen]) * 10000
            lower = np.array([float(r["lower"]) for r in chosen]) * 10000
            upper = np.array([float(r["upper"]) for r in chosen]) * 10000
            positions = np.arange(4) + offset
            ax.hlines(positions, lower, upper, color=color, linewidth=1.6)
            ax.scatter(estimate, positions, color=color, marker=marker, s=28, label=label)
        ax.axvline(0, color="#333333", linestyle=":", linewidth=1)
        ax.set_title(family.upper() if family != "dlinear" else "DLinear")
        ax.set_yticks(
            range(4),
            [
                f"{months[starts[row['fold']].month - 1]} {starts[row['fold']].year}"
                f" · {row['sessions']} sesiones"
                for row in chosen
            ],
        )
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylim(3.5, -0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.supxlabel("Δ MAE frente a cero (×10⁻⁴). Un valor negativo indica menor error.", y=0.065)
    fig.suptitle("Evaluación real por periodo · media de errores de tres semillas", y=0.93)
    fig.text(
        0.5,
        0.02,
        "Intervalos exploratorios del 95 %, bloques de 5 sesiones, 2.000 réplicas.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.9))
    save(fig, directory, "predictive-periods")


def financial(directory):
    records = rows(directory / "financial-aggregates.csv")
    groups = defaultdict(list)
    for row in records:
        if row["cost_bps"] in {"10", "10.0"}:
            groups[(row["family"], row["variant"])].append(row)
    variants = (
        "cash",
        "hold_initial",
        "rebalance_50",
        "rebalance_100",
        "ppo",
        "double_dqn",
        "ppo_window",
        "ppo_gru",
        "ppo_hmm",
        "ppo_episodic",
        "ppo_episodic_hmm",
        "ppo_recent_aux",
        "ppo_replay_aux",
    )
    families = sorted(family for family, variant in groups if family != "all" and variant == "ppo")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharex=True)
    for ax, labels, keys in (
        (axes[0], variants, [("all", v) for v in variants]),
        (axes[1], families, [(f, "ppo") for f in families]),
    ):
        for position, key in enumerate(keys):
            group = groups[key]
            expected = 1 if group[0]["kind"] == "control" else 3
            if len(group) != expected or any(
                int(row["worlds"]) != (512 if key[0] == "all" else 64) for row in group
            ):
                raise ValueError("La figura financiera no tiene la población esperada")
            values = [float(row["mean_net_return"]) * 100 for row in group]
            mean = math.fsum(values) / len(values)
            ax.barh(position, mean, color="#245f9e" if expected == 3 else "#969696", height=0.58)
            ax.scatter(
                values,
                [position] * len(values),
                facecolors="white",
                edgecolors="#333333",
                s=20,
                linewidth=0.8,
                zorder=3,
            )
        ax.set_yticks(range(len(labels)), labels)
        ax.invert_yaxis()
        ax.axvline(0, color="#333333", linewidth=1)
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlabel("Retorno neto medio (%)")
    axes[0].set_title("Variantes y controles · 512 mundos")
    axes[1].set_title("PPO por familia · 64 mundos")
    fig.suptitle("Auditoría sintética · costes de 10 puntos básicos")
    fig.text(
        0.5,
        0.015,
        "Puntos: medias por semilla. Mismos mundos y costes, sin intervalos ni anualización.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    save(fig, directory, "financial-scenarios")


def save(figure, directory, name):
    svg = directory / f"{name}.svg"
    figure.savefig(svg, metadata={"Date": None, "Creator": "Matplotlib"})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    figure.savefig(directory / f"{name}.png", dpi=150, metadata={"Software": "Matplotlib"})
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictive", "financial", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    safe_destination(args.output)
    if args.output.exists():
        parser.error("La salida debe ser nueva")
    receipts = {}
    for kind, source in (("predictive", args.predictive), ("financial", args.financial)):
        outside_source(source, args.output)
        outside_source(args.output, source)
        report, signature = read_manifest(source / "comparison.json", 16 * 1024**2)
        if report.get("status") != "completed" or report.get("final_test_opened") is not False:
            raise ValueError("La comparación no está completa o abrió la reserva")
        files = (
            ("cases.csv", "methods.csv", "folds.csv", "intervals.csv")
            if kind == "predictive"
            else ("aggregates.csv", "paired_summary.csv")
        )
        for name in files:
            digest = report["artifacts"][name]
            if isinstance(digest, dict):
                digest = digest["sha256"]
            if sha256(source / name) != digest:
                raise ValueError("Un agregado ha cambiado desde su comprobación")
        receipts[kind] = dict(
            report_sha256=signature, counts=report["counts"], provenance=report["provenance"]
        )
        if kind == "predictive":
            receipts[kind].update(
                method=report["method"],
                versions=report["versions"],
                analysis_source_sha256=report["analysis_source_sha256"],
            )
    args.output.mkdir(parents=True, exist_ok=False)
    for kind, source, names in (
        ("predictive", args.predictive, ("cases.csv", "methods.csv", "folds.csv", "intervals.csv")),
        ("financial", args.financial, ("aggregates.csv", "paired_summary.csv")),
    ):
        for name in names:
            with (source / name).open(encoding="utf-8", newline="") as stream:
                with (args.output / f"{kind}-{name}").open(
                    "w", encoding="utf-8", newline=""
                ) as target:
                    csv.writer(target, lineterminator="\n").writerows(csv.reader(stream))
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "mars-titan"}
    )
    predictions(args.output, receipts["predictive"]["provenance"]["folds"])
    financial(args.output)
    receipts["artifacts"] = {p.name: sha256(p) for p in sorted(args.output.iterdir())}
    receipts["renderer"] = {
        "matplotlib": matplotlib.__version__,
        "script_sha256": sha256(Path(__file__)),
        "csv_line_endings": "LF",
    }
    atomic_json(args.output / "evidence.json", receipts)


if __name__ == "__main__":
    main()
