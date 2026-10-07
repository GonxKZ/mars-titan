"""Exportar tablas agregadas verificadas y figuras de las campañas comparativas."""

import argparse
import csv
import hashlib
import io
import math
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.macro_coverage import _publish_directory
from mars_titan.data.storage import atomic_json, outside_source, sha256

MAX_CSV_BYTES = 32 * 1024**2
MAX_ROWS = 100_000
FILES = {
    "predictive": ("cases.csv", "methods.csv", "folds.csv", "intervals.csv"),
    "financial": ("aggregates.csv", "paired_summary.csv"),
    "reliability": ("cases.csv",),
}
IDENTITY = ("stage", "phase", "family", "method", "seed", "included", "primary", "parent_id")


def rows(path, expected=None):
    safe_destination(path)
    if not path.is_file():
        raise ValueError("El agregado no es un archivo regular")
    with path.open("rb") as stream:
        payload = stream.read(MAX_CSV_BYTES + 1)
    if len(payload) > MAX_CSV_BYTES:
        raise ValueError("El agregado supera el límite de bytes")
    if expected is not None and hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Un agregado ha cambiado desde su comprobación")
    reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""), strict=True)
    fields = reader.fieldnames or []
    if not fields or len(fields) > 256 or len(set(fields)) != len(fields) or "" in fields:
        raise ValueError("El agregado tiene columnas ausentes o duplicadas")
    records = []
    for row in reader:
        if None in row or None in row.values() or len(records) >= MAX_ROWS:
            raise ValueError("El agregado tiene filas inválidas o supera el límite")
        records.append(row)
    if not records:
        raise ValueError("El agregado está vacío")
    return records


def write_csv(path, records):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        stream.flush()
        os.fsync(stream.fileno())


def number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Una métrica del agregado no es finita")
    return result


def valid_hash(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def match_summary(records, summary, identity):
    """Contrastar las copias CSV y JSON sin imponer el orden de sus filas."""
    if len(records) != len(summary):
        raise ValueError("El resumen JSON no corresponde a su CSV")
    fields = set(records[0])
    expected = {}
    for original in summary:
        if set(original) - fields:
            raise ValueError("El resumen JSON tiene columnas ausentes del CSV")
        row = {name: "" if original.get(name) is None else str(original[name]) for name in fields}
        key = tuple(row[name] for name in identity)
        if key in expected:
            raise ValueError("El resumen JSON contiene un identificador duplicado")
        expected[key] = row
    actual = {tuple(row[name] for name in identity): row for row in records}
    if len(actual) != len(records) or actual != expected:
        raise ValueError("El resumen JSON no corresponde a su CSV")


def comparison_markets(method):
    if "market_stratification" not in method:
        return (None,)
    if method["market_stratification"] != "separate_markets" or method.get("markets") != [
        "CN",
        "US",
    ]:
        raise ValueError("La comparación conjunta debe separar explícitamente US y CN")
    return ("CN", "US")


def validate_predictive(report, tables, folds):
    markets = comparison_markets(report["method"])
    extra = ("market",) if markets != (None,) else ()
    for fold in folds:
        if extra or "markets" in fold or "market_counts" in fold:
            if (
                not extra
                or fold.get("markets") != list(markets)
                or not isinstance(fold.get("market_counts"), dict)
                or set(fold["market_counts"]) != set(markets)
            ):
                raise ValueError("La separación por mercado no corresponde a la procedencia")
    aggregate_key = ("partition", "family", "method", *extra)
    for field, filename, identity in (
        ("overall", "methods.csv", aggregate_key),
        ("by_fold", "folds.csv", ("fold", *aggregate_key)),
        ("intervals", "intervals.csv", ("fold", *aggregate_key, "reference", "block_length")),
    ):
        match_summary(tables[filename], report[field], identity)
    identifiers = {fold["id"]: fold for fold in folds}
    metadata = {(row["fold"], row["id"]): row["metadata"]["case"] for row in report["metadata"]}
    if len(metadata) != len(report["metadata"]):
        raise ValueError("La metadata contiene un modelo duplicado")
    cases, models, groups = {}, defaultdict(dict), defaultdict(list)
    for row in tables["cases.csv"]:
        key = (row["fold"], row["id"], row["partition"])
        if (
            key in cases
            or key[0] not in identifiers
            or not key[1]
            or key[2] not in {"calibration", "evaluation"}
            or row["included"] not in {"True", "False"}
            or not valid_hash(row["predictions_sha256"])
            or not 1 <= int(row["sessions"]) <= int(row["samples"])
        ):
            raise ValueError("Un caso predictivo está duplicado o no pertenece a la población")
        counts = identifiers[key[0]].get("counts", {})
        if key[2] in counts and int(row["samples"]) != counts[key[2]]:
            raise ValueError("Las muestras del caso no corresponden a su cohorte temporal")
        if key[:2] not in metadata:
            raise ValueError("El modelo no aparece en los metadatos congelados")
        seed = metadata[key[:2]].get("seed")
        if row["seed"] != ("" if seed is None else str(seed)):
            raise ValueError("La semilla del caso no coincide con los metadatos congelados")
        for field in ("checkpoint_sha256", "source_report_sha256"):
            if field in row and not valid_hash(row[field]):
                raise ValueError("La huella del estado congelado no es válida")
        cases[key] = row
        models[key[:2]][key[2]] = row
        if extra:
            declared = identifiers[key[0]].get("market_counts", {})
            if (
                set(declared) != set(markets)
                or sum(int(row.get(f"samples_{market}", 0)) for market in markets)
                != int(row["samples"])
                or sum(int(row.get(f"sessions_{market}", 0)) for market in markets)
                != int(row["sessions"])
            ):
                raise ValueError("Los recuentos por mercado no concilian con el caso completo")
        for market in markets:
            member = row
            if market is not None:
                samples, sessions = int(row[f"samples_{market}"]), int(row[f"sessions_{market}"])
                if not 1 <= sessions <= samples or samples != declared[market].get(key[2]):
                    raise ValueError("Un mercado no conserva su población y sesiones")
                member = dict(row, samples=str(samples), sessions=str(sessions), market=market)
            if row["included"] == "True":
                groups[(row["fold"], *(member[name] for name in aggregate_key))].append(member)
    if (
        len(cases) != report["counts"]["prediction_files"]
        or len(models) != report["counts"]["models"]
        or set(models) != set(metadata)
    ):
        raise ValueError("Los conteos predictivos no coinciden con los casos")
    for pair in models.values():
        if set(pair) != {"calibration", "evaluation"}:
            raise ValueError("Falta una partición del modelo congelado")
        left, right = pair["calibration"], pair["evaluation"]
        if any(left[name] != right[name] for name in IDENTITY) or left.get(
            "checkpoint_sha256"
        ) != right.get("checkpoint_sha256"):
            raise ValueError("Las particiones no pertenecen al mismo modelo congelado")
    by_fold, overall = {}, defaultdict(list)
    for row in tables["folds.csv"]:
        key = tuple(row[name] for name in ("fold", *aggregate_key))
        members = groups.get(key, [])
        if (
            key in by_fold
            or len(members) != int(row["models"])
            or not members
            or {member["sessions"] for member in members} != {row["sessions"]}
            or len({member["seed"] for member in members}) != len(members)
        ):
            raise ValueError("Un agregado por ventana no corresponde a sus modelos")
        by_fold[key] = row
        overall[key[1:]].extend(members)
    if set(by_fold) != set(groups):
        raise ValueError("Faltan agregados de la población seleccionada")
    seen = set()
    for row in tables["methods.csv"]:
        key = tuple(row[name] for name in aggregate_key)
        members = overall.get(key, [])
        if (
            key in seen
            or not members
            or int(row["models"]) != len(members)
            or int(row["folds"]) != len({member["fold"] for member in members})
        ):
            raise ValueError("Un agregado por método no corresponde a sus ventanas")
        seen.add(key)
    if seen != set(overall):
        raise ValueError("Faltan agregados por método")
    for row in tables["intervals.csv"]:
        key = tuple(row[name] for name in ("fold", *aggregate_key))
        if key not in by_fold or any(
            row[name] != by_fold[key][name] for name in ("models", "sessions")
        ):
            raise ValueError("Un intervalo no corresponde al agregado de su ventana")
    return cases


def validate_reliability(report, records, predictive, cases):
    method = report["method"]
    markets = comparison_markets(predictive["method"])
    extra = ("market",) if markets != (None,) else ()
    if (
        report.get("kind") != "frozen_campaign_reliability"
        or report.get("target_kind") != "residual_return"
        or report.get("coverage_guaranteed") is not False
        or method.get("confidence_levels") != [0.9, 0.95]
        or method.get("calibration_partition") != "calibration"
        or method.get("evaluation_partition") != "evaluation"
        or method.get("selection") != "frozen_before_calibration"
        or method.get("seed_or_fold_pooling") is not False
        or (
            extra
            and (
                method.get("calibration_grouping") != "model_fold_seed_market"
                or method.get("market_pooling") is not False
            )
        )
    ):
        raise ValueError("El recibo de fiabilidad no acredita el diagnóstico residual congelado")
    for field in ("reference_sha256", "completion_sha256", "folds", "final_test_opened"):
        if report["provenance"].get(field) != predictive["provenance"].get(field):
            raise ValueError("La fiabilidad pertenece a otra campaña o procedencia temporal")
    if (
        report["counts"]["models"] != predictive["counts"]["models"]
        or report["counts"]["prediction_files"] != len(cases)
        or report["counts"]["folds"] != len(predictive["provenance"]["folds"])
    ):
        raise ValueError("La fiabilidad no incluye la misma población de modelos")
    match_summary(records, report["cases"], ("fold", "id", "prediction_kind", "confidence", *extra))
    folds = {fold["id"]: fold for fold in predictive["provenance"]["folds"]}
    seen = set()
    for row in records:
        key = (
            row["fold"],
            row["id"],
            row["prediction_kind"],
            row["confidence"],
            *(row[name] for name in extra),
        )
        if (
            key in seen
            or key[2:4] not in {("point", ""), ("interval", "0.9"), ("interval", "0.95")}
            or (extra and row["market"] not in markets)
            or row["target_kind"] != "residual_return"
            or row["coverage_guaranteed"] != "False"
        ):
            raise ValueError(
                "La fiabilidad tiene un caso duplicado o una interpretación incompatible"
            )
        seen.add(key)
        for partition in ("calibration", "evaluation"):
            case = cases.get((*key[:2], partition))
            if case is None or any(row[name] != case[name] for name in IDENTITY):
                raise ValueError("El diagnóstico no corresponde al mismo modelo")
            if not valid_hash(case.get("checkpoint_sha256")) or not valid_hash(
                case.get("source_report_sha256")
            ):
                raise ValueError(
                    "La comparación antigua requiere regenerarse con las huellas "
                    "del checkpoint y del recibo"
                )
            scoped = (
                dict(
                    case,
                    **{name: case[f"{name}_{row['market']}"] for name in ("samples", "sessions")},
                )
                if extra
                else case
            )
            if row["checkpoint_sha256"] != case["checkpoint_sha256"] or any(
                row[f"{partition}_{name}"] != scoped[name]
                for name in ("predictions_sha256", "source_report_sha256", "samples", "sessions")
            ):
                raise ValueError("La fiabilidad no corresponde al checkpoint o a sus predicciones")
            if [row[f"{partition}_start"], row[f"{partition}_end"]] != folds[key[0]]["windows"][
                partition
            ]:
                raise ValueError("La fiabilidad no corresponde al mismo corte temporal")
    expected = {
        (fold, identifier, kind, confidence, *((market,) if extra else ()))
        for fold, identifier, partition in cases
        if partition == "evaluation"
        for kind, confidence in (("point", ""), ("interval", "0.9"), ("interval", "0.95"))
        for market in markets
    }
    if seen != expected:
        raise ValueError("Faltan diagnósticos puntuales o de intervalo")


def validate_financial(report, tables):
    population = report["population"]
    families = population["worlds_by_family"]
    seeds = {str(seed) for seed in population["policy_seeds"]}
    costs = {number(cost) for cost in population["cost_bps"]}
    if (
        report.get("domain") != "synthetic"
        or report.get("split") != "audit"
        or not families
        or "all" in families
        or any(type(count) is not int or count < 1 for count in families.values())
        or sum(families.values()) != population["worlds"]
        or not seeds
        or len(seeds) != len(population["policy_seeds"])
        or len(costs) != len(population["cost_bps"])
        or 10 not in costs
    ):
        raise ValueError("La población financiera no corresponde a una auditoría sintética")
    worlds = {"all": population["worlds"], **families}
    groups = defaultdict(list)
    for row in tables["aggregates.csv"]:
        key = (row["variant"], row["family"], number(row["cost_bps"]))
        if (
            key[1] not in worlds
            or key[2] not in costs
            or not key[0]
            or row["kind"] not in {"control", "learned"}
            or int(row["worlds"]) != worlds[key[1]]
        ):
            raise ValueError("Un agregado financiero tiene mundos, costes o familia incompatibles")
        for field, value in row.items():
            if field.startswith("mean_"):
                number(value)
        groups[key].append(row)
    variants = {key[0] for key in groups}
    # Cada grupo ya pertenece al producto. La cardinalidad comprueba que está completo.
    if len(groups) != len(variants) * len(worlds) * len(costs):
        raise ValueError("Faltan agregados financieros de una familia o coste")
    for group in groups.values():
        expected = {""} if group[0]["kind"] == "control" else seeds
        if (
            len(group) != len(expected)
            or {row["seed"] for row in group} != expected
            or len({row["kind"] for row in group}) != 1
        ):
            raise ValueError("Las semillas financieras están duplicadas o son incompletas")
    seen = set()
    for row in tables["paired_summary.csv"]:
        cost = number(row["cost_bps"])
        key = (row["variant"], row["control"], row["family"], cost)
        learned = groups.get((key[0], key[2], cost), [])
        control = groups.get((key[1], key[2], cost), [])
        if (
            key in seen
            or not learned
            or not control
            or learned[0]["kind"] != "learned"
            or control[0]["kind"] != "control"
            or int(row["worlds"]) != worlds[key[2]]
            or int(row["policy_seeds"]) != len(seeds)
        ):
            raise ValueError("Un contraste financiero no corresponde a las poblaciones emparejadas")
        seen.add(key)
        for field, value in row.items():
            if field.startswith("delta_"):
                number(value)


def windows(provenance):
    folds = provenance["folds"]
    if (
        provenance.get("final_test_opened") is not False
        or not 1 <= len(folds) <= 128
        or not all(
            valid_hash(provenance.get(field)) for field in ("reference_sha256", "completion_sha256")
        )
    ):
        raise ValueError("La procedencia temporal no es válida")
    identifiers = set()
    previous_end = None
    for fold in folds:
        identifier = fold["id"]
        if not isinstance(identifier, str) or not identifier or identifier in identifiers:
            raise ValueError("Las ventanas tienen identificadores inválidos o duplicados")
        identifiers.add(identifier)
        calibration = [date.fromisoformat(value) for value in fold["windows"]["calibration"]]
        evaluation = [date.fromisoformat(value) for value in fold["windows"]["evaluation"]]
        if (
            len(calibration) != 2
            or len(evaluation) != 2
            or not calibration[0] < calibration[1] <= evaluation[0] < evaluation[1]
            or evaluation[1] > date(2024, 1, 1)
            or (previous_end is not None and evaluation[0] < previous_end)
        ):
            raise ValueError("Las ventanas no respetan el orden temporal o la reserva cerrada")
        previous_end = evaluation[1]
    return folds


def predictive_series(records, folds, method, *, market=None):
    markets = comparison_markets(method)
    if market not in markets:
        raise ValueError("La serie necesita un mercado declarado por la comparación")
    if markets != (None,):
        if {row.get("market") for row in records} != set(markets):
            raise ValueError("Los intervalos no contienen los dos mercados declarados")
        records = [row for row in records if row["market"] == market]
    if (
        not 0 < number(method["confidence"]) < 1
        or 5 not in method["block_lengths"]
        or type(method["repetitions"]) is not int
        or method["repetitions"] < 1
    ):
        raise ValueError("El método no acredita los intervalos representados")
    identifiers = [fold["id"] for fold in folds]
    groups = defaultdict(dict)
    seen = set()
    for row in records:
        key = tuple(
            row[name]
            for name in ("fold", "partition", "family", "method", "reference", "block_length")
        )
        if key in seen or row["fold"] not in identifiers:
            raise ValueError("El intervalo tiene una ventana desconocida o está duplicado")
        seen.add(key)
        if row["partition"] not in {"calibration", "evaluation"}:
            raise ValueError("El intervalo no pertenece a una partición admitida")
        if (row["partition"], row["reference"], row["block_length"]) == ("evaluation", "zero", "5"):
            if (
                int(row["repetitions"]) != method["repetitions"]
                or int(row["seed"]) != method["seed"]
            ):
                raise ValueError("Las réplicas o la semilla no coinciden con el método del recibo")
            if int(row["sessions"]) < 1 or int(row["models"]) < 1:
                raise ValueError("El intervalo no tiene población")
            number(row["estimate"])
            if bool(row["lower"]) != bool(row["upper"]):
                raise ValueError("El intervalo está incompleto")
            if row["lower"]:
                if number(row["lower"]) > number(row["upper"]):
                    raise ValueError("Los límites del intervalo están invertidos")
            elif not row["reason"]:
                raise ValueError("Falta la razón del intervalo indefinido")
            groups[(row["family"], row["method"])][row["fold"]] = row
    result = {}
    for family in ("rnn", "lstm", "gru", "dlinear"):
        for name in ("reference", "neural_mae", "neural_mse"):
            group = groups[(family, name)]
            if set(group) != set(identifiers):
                raise ValueError("La serie no contiene exactamente las ventanas de procedencia")
            result[(family, name)] = [group[identifier] for identifier in identifiers]
        for index in range(len(folds)):
            if (
                len(
                    {
                        result[(family, name)][index]["sessions"]
                        for name in ("reference", "neural_mae", "neural_mse")
                    }
                )
                != 1
            ):
                raise ValueError("Las series comparadas no tienen las mismas sesiones")
    return result


def predictions(directory, folds, method_metadata, records=None):
    records = rows(directory / "predictive-intervals.csv") if records is None else records
    markets = comparison_markets(method_metadata)
    series_by_market = {
        market: predictive_series(records, folds, method_metadata, market=market)
        for market in markets
    }
    count = len(folds)

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
    fig, axes = plt.subplots(
        2 * len(markets),
        2,
        figsize=(11, len(markets) * max(7, 2 + 0.8 * count)),
        sharex=True,
        sharey=len(markets) == 1,
    )
    panels = [
        (market, family) for market in markets for family in ("rnn", "lstm", "gru", "dlinear")
    ]
    for ax, (market, family) in zip(axes.flat, panels, strict=True):
        series = series_by_market[market]
        for offset, (method, label, color, marker) in zip((-0.2, 0, 0.2), methods, strict=True):
            chosen = series[(family, method)]
            estimate = np.array([float(r["estimate"]) for r in chosen]) * 10000
            lower = np.array([float(r["lower"]) if r["lower"] else np.nan for r in chosen]) * 10000
            upper = np.array([float(r["upper"]) if r["upper"] else np.nan for r in chosen]) * 10000
            positions = np.arange(count) + offset
            defined = np.isfinite(lower)
            ax.hlines(
                positions[defined], lower[defined], upper[defined], color=color, linewidth=1.6
            )
            ax.scatter(estimate, positions, color=color, marker=marker, s=28, label=label)
        ax.axvline(0, color="#333333", linestyle=":", linewidth=1)
        title = family.upper() if family != "dlinear" else "DLinear"
        ax.set_title(f"{market} · {title}" if market is not None else title)
        ax.set_yticks(
            range(count),
            [
                f"{months[starts[row['fold']].month - 1]} {starts[row['fold']].year}"
                f" · {row['sessions']} sesiones"
                for row in chosen
            ],
        )
        ax.grid(axis="x", color="#dddddd", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(count - 0.5, -0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.supxlabel("Δ MAE frente a cero (×10⁻⁴). Un valor negativo indica menor error.", y=0.065)
    fig.suptitle("Evaluación por periodo · media de errores por semilla", y=0.93)
    undefined = any(
        not row["lower"]
        for series in series_by_market.values()
        for group in series.values()
        for row in group
    )
    fig.text(
        0.5,
        0.02,
        f"Intervalos exploratorios del {number(method_metadata['confidence']) * 100:g} %, "
        f"bloques de 5 sesiones, {method_metadata['repetitions']} réplicas."
        + (" Sin barra si el intervalo está indefinido." if undefined else ""),
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.9))
    save(fig, directory, "predictive-periods")


def financial(directory, population, records=None):
    records = rows(directory / "financial-aggregates.csv") if records is None else records
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
            if not group:
                raise ValueError("Falta una variante financiera")
            control = group[0]["kind"] == "control"
            seeds = {""} if control else {str(seed) for seed in population["policy_seeds"]}
            worlds = (
                population["worlds"] if key[0] == "all" else population["worlds_by_family"][key[0]]
            )
            if (
                len(group) != len(seeds)
                or {row["seed"] for row in group} != seeds
                or any(int(row["worlds"]) != worlds for row in group)
            ):
                raise ValueError("La figura financiera no tiene la población esperada")
            values = [number(row["mean_net_return"]) * 100 for row in group]
            mean = math.fsum(values) / len(values)
            ax.barh(position, mean, color="#969696" if control else "#245f9e", height=0.58)
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
    axes[0].set_title(f"Variantes y controles · {population['worlds']} mundos")
    axes[1].set_title("PPO por familia")
    axes[1].set_yticks(
        range(len(families)),
        [f"{family} · {population['worlds_by_family'][family]} mundos" for family in families],
    )
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


def load_source(kind, source, output):
    safe_destination(source)
    outside_source(source, output)
    outside_source(output, source)
    name = "reliability.json" if kind == "reliability" else "comparison.json"
    report, signature = read_manifest(source / name, 64 * 1024**2)
    if report.get("status") != "completed" or report.get("final_test_opened") is not False:
        raise ValueError("La comparación no está completa o abrió la reserva")
    tables, hashes = {}, {}
    for name in FILES[kind]:
        digest = report["artifacts"][name]
        if isinstance(digest, dict):
            digest = digest["sha256"]
        if not valid_hash(digest):
            raise ValueError("Falta la huella SHA-256 del agregado")
        tables[name] = rows(source / name, digest)
        hashes[name] = digest
    receipt = dict(
        report_sha256=signature,
        counts=report["counts"],
        provenance=report["provenance"],
        source_artifacts=hashes,
        final_test_opened=False,
    )
    for field in (
        "method",
        "versions",
        "analysis_source_sha256",
        "population",
        "target_kind",
        "coverage_guaranteed",
        "uncertainty",
    ):
        if field in report:
            receipt[field] = report[field]
    return report, receipt, tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("predictive", "financial", "reliability", "output"):
        parser.add_argument(f"--{name}", type=Path, required=name in {"predictive", "output"})
    args = parser.parse_args(argv)
    safe_destination(args.output)
    if args.output.exists():
        parser.error("La salida debe ser nueva")
    reports, receipts, tables = {}, {}, {}
    for kind in FILES:
        source = getattr(args, kind)
        if source is not None:
            reports[kind], receipts[kind], tables[kind] = load_source(kind, source, args.output)
    folds = windows(reports["predictive"]["provenance"])
    cases = validate_predictive(reports["predictive"], tables["predictive"], folds)
    if "reliability" in reports:
        validate_reliability(
            reports["reliability"], tables["reliability"]["cases.csv"], reports["predictive"], cases
        )
        receipts["reliability"]["linkage"] = "checkpoint_predictions_and_source_reports"
    if "financial" in reports:
        validate_financial(reports["financial"], tables["financial"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".campaign-export-", dir=args.output.parent) as temporary:
        stage = Path(temporary) / "export"
        stage.mkdir()
        try:
            for kind, source_tables in tables.items():
                for name, records in source_tables.items():
                    write_csv(stage / f"{kind}-{name}", records)
            with plt.rc_context(
                {"font.family": "DejaVu Sans", "font.size": 10, "svg.hashsalt": "mars-titan"}
            ):
                predictions(
                    stage,
                    folds,
                    reports["predictive"]["method"],
                    tables["predictive"]["intervals.csv"],
                )
                if "financial" in reports:
                    financial(
                        stage,
                        reports["financial"]["population"],
                        tables["financial"]["aggregates.csv"],
                    )
            receipts["artifacts"] = {p.name: sha256(p) for p in sorted(stage.iterdir())}
            receipts["renderer"] = dict(
                matplotlib=matplotlib.__version__,
                script_sha256=sha256(Path(__file__)),
                csv_line_endings="LF",
            )
            atomic_json(stage / "evidence.json", receipts)
            for path in stage.iterdir():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            _publish_directory(stage, args.output)
            descriptor = os.open(args.output.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        finally:
            plt.close("all")


if __name__ == "__main__":
    main()
