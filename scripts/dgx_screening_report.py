"""Informe de la criba de una campaña DGX: IC por sesión, contrastes pareados y combinaciones.

Lee de un estado de `comparison_campaign` las predicciones emitidas por cada modelo terminado en
uno o varios orígenes y una semilla (validación, calibración y evaluación) y escribe un informe
JSON y otro Markdown con:

- el IC de rangos por sesión de cada modelo y tramo (la métrica de selección de la campaña: media
  de la correlación de Spearman entre predicción y objetivo en cada sesión con al menos 30
  activos; una sesión es un mercado en un instante), por mercado si hay varios, su ICIR y la
  desviación de su curva de validación por época;
- el contraste pareado de cada variante con su control: la diferencia de IC sesión a sesión, que
  cancela el ruido común de las sesiones, con su media, su error típico y su t. Se decide solo con
  validación y calibración; la evaluación se informa como evidencia fuera de muestra del origen.
  Con varios orígenes, el contraste agregado reúne las sesiones de todos;
- combinaciones de modelos por media de rangos dentro de cada sesión, elegidas hacia delante solo
  con validación y calibración;
- con `--compare`, la diferencia entre cada modelo de este estado y el mismo modelo de otro (la
  campaña conjunta frente a la de EE. UU.) en sus sesiones comunes.

Solo lee: no toca los estados ni abre ningún tramo que la campaña no haya emitido.

Uso:

    python scripts/dgx_screening_report.py --state ESTADO --year 2013 --seed 42 \
        --controls CONTROLES.json --output CARPETA [--reference OTRO] [--compare OTRO]
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SPLITS = ("validation", "calibration", "evaluation")
DECISION = ("validation", "calibration")
MIN_ASSETS = 30
COLUMNS = ["sample_id", "market", "prediction_at", "target", "prediction"]


def average_ranks(values):
    """Rangos medios (los empates comparten el rango medio), como en la correlación de Spearman."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(1, len(values) + 1, dtype=np.float64)
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], len(values)]
    for start, end in zip(starts, ends, strict=True):
        if end - start > 1:
            ranks[order[start:end]] = (start + 1 + end) / 2
    return ranks


def correlation(left, right):
    left = left - left.mean()
    right = right - right.mean()
    scale = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right) / scale if scale > 0 else float("nan")


def sample_digest(column):
    """Huella del orden de las muestras de un tramo: dos modelos se combinan si coincide."""
    digest = hashlib.sha256()
    for chunk in column.chunks:
        for buffer in chunk.buffers():
            if buffer is not None:
                digest.update(memoryview(buffer))
    return digest.hexdigest()


def load_split(run, split):
    """Predicciones de un tramo ordenadas por mercado, instante y muestra, con sus sesiones."""
    path = run / f"{split}-predictions.parquet"
    if not path.is_file():
        return None
    table = pq.read_table(path, columns=COLUMNS).sort_by(
        [("market", "ascending"), ("prediction_at", "ascending"), ("sample_id", "ascending")]
    )
    market = table.column("market").combine_chunks().dictionary_encode()
    codes = market.indices.to_numpy(zero_copy_only=False)
    instant = table.column("prediction_at").cast("int64").to_numpy()
    change = (instant[1:] != instant[:-1]) | (codes[1:] != codes[:-1])
    starts = np.r_[0, np.flatnonzero(change) + 1]
    names = market.dictionary.to_pylist()
    bounds = list(zip(starts, np.r_[starts[1:], len(instant)], strict=True))
    return dict(
        sample=sample_digest(table.column("sample_id")),
        target=table.column("target").to_numpy(),
        prediction=table.column("prediction").to_numpy(),
        sessions=[(names[codes[start]], int(instant[start]), start, end) for start, end in bounds],
    )


def session_ic(data, prediction=None):
    """IC de rangos de cada sesión con suficientes activos: {(mercado, instante): IC}."""
    prediction = data["prediction"] if prediction is None else prediction
    result = {}
    for market, instant, start, end in data["sessions"]:
        if end - start < MIN_ASSETS:
            continue
        value = correlation(
            average_ranks(prediction[start:end]), average_ranks(data["target"][start:end])
        )
        if not math.isnan(value):
            result[(market, instant)] = value
    return result


def summary(values):
    values = np.asarray(list(values), dtype=np.float64)
    if len(values) == 0:
        return dict(sessions=0, mean=None, std=None, icir=None)
    std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    return dict(
        sessions=int(len(values)),
        mean=float(values.mean()),
        std=std,
        icir=float(values.mean() / std) if std > 0 else None,
    )


def paired(candidate, control):
    """Diferencia de IC sesión a sesión entre dos modelos sobre sus sesiones comunes."""
    keys = sorted(set(candidate) & set(control))
    if len(keys) < 2:
        return dict(sessions=len(keys), mean=None, se=None, t=None)
    difference = np.asarray([candidate[key] - control[key] for key in keys])
    se = float(difference.std(ddof=1) / math.sqrt(len(keys)))
    return dict(
        sessions=len(keys),
        mean=float(difference.mean()),
        se=se,
        t=float(difference.mean() / se) if se > 0 else None,
    )


def pooled(ics, splits=DECISION):
    """Sesiones de varios tramos juntas. Con los de decisión es lo único con lo que se elige."""
    merged = {}
    for split in splits:
        merged.update({(split, *key): value for key, value in ics[split].items()})
    return merged


def epoch_curve(run):
    """IC de rangos de validación por época, del informe del ajuste."""
    path = run / "fit" / "run.json"
    if not path.is_file():
        return []
    history = json.loads(path.read_text()).get("history") or []
    return [
        (int(entry["epoch"]), float(entry["validation"]["session_rank_ic"]))
        for entry in history
        if (entry.get("validation") or {}).get("session_rank_ic") is not None
    ]


def completed(state, year, seed):
    """Registros de la cola de los trabajos terminados de un origen y una semilla."""
    records = {}
    for path in sorted((state / "queue").glob(f"*--{year}--seed-{seed}.json")):
        record = json.loads(path.read_text())
        if record.get("status") == "completed":
            records[record["model"]] = record
    return records


def model_report(state, record, year, seed):
    run = state / "runs" / record["model"] / str(year) / f"seed-{seed}"
    splits = {split: load_split(run, split) for split in SPLITS}
    if any(data is None for data in splits.values()):
        return None
    ics = {split: session_ic(data) for split, data in splits.items()}
    curve = epoch_curve(run)
    late = [value for epoch, value in curve if epoch >= 2]
    attempt = (record.get("attempts") or [{}])[-1]
    result = record.get("result") or {}
    markets = sorted({key[0] for values in ics.values() for key in values})
    by_market = {
        market: {
            split: summary(v for key, v in ics[split].items() if key[0] == market)
            for split in SPLITS
        }
        for market in markets
    }
    score = result.get("best_score")
    return dict(
        model=record["model"],
        tier=record.get("tier"),
        state=str(state),
        best_epoch=result.get("best_epoch"),
        selected_validation_ic=None if score is None else 1 - 2 * score,
        hours=None if attempt.get("seconds") is None else attempt["seconds"] / 3600,
        epochs=curve,
        epoch_std=float(np.std(late, ddof=1)) if len(late) > 1 else None,
        ic={split: summary(values.values()) for split, values in ics.items()},
        decision_ic=summary(pooled(ics).values())["mean"],
        ic_by_market=by_market if len(markets) > 1 else {},
        _ics=ics,
        _splits=splits,
        _ranks={},
    )


def normalized_ranks(report, split):
    """Rangos de las predicciones dentro de cada sesión, entre 0 y 1. Se calculan una vez."""
    if split not in report["_ranks"]:
        data = report["_splits"][split]
        ranks = np.zeros(len(data["target"]), dtype=np.float64)
        for _, _, start, end in data["sessions"]:
            ranks[start:end] = average_ranks(data["prediction"][start:end]) / (end - start)
        report["_ranks"][split] = ranks
    return report["_ranks"][split]


def blend(members, split):
    """IC por sesión de la media de rangos de varios modelos, o None si no comparten muestras."""
    base = members[0]["_splits"][split]
    if any(member["_splits"][split]["sample"] != base["sample"] for member in members[1:]):
        return None
    total = sum(normalized_ranks(member, split) for member in members)
    return session_ic(base, total / len(members))


def forward_selection(reports, limit):
    """Combinación elegida hacia delante por el IC de validación más calibración."""
    chosen, steps, best = [], [], None
    while len(chosen) < limit:
        candidates = []
        for name in sorted(reports):
            if name in chosen:
                continue
            members = [reports[item] for item in (*chosen, name)]
            blends = {split: blend(members, split) for split in DECISION}
            if any(item is None for item in blends.values()):
                continue
            candidates.append((summary(pooled(blends).values())["mean"], name))
        if not candidates:
            break
        score, name = max(candidates)
        if score is None or (best is not None and score <= best):
            break
        best = score
        chosen.append(name)
        members = [reports[item] for item in chosen]
        evaluation = blend(members, "evaluation")
        steps.append(
            dict(
                members=list(chosen),
                decision_ic=score,
                evaluation=summary(evaluation.values()) if evaluation is not None else None,
            )
        )
    return steps


def contrast(name, control, group, candidate, base):
    return dict(
        model=name,
        control=control,
        group=group,
        decision=paired(pooled(candidate["_ics"]), pooled(base["_ics"])),
        validation=paired(candidate["_ics"]["validation"], base["_ics"]["validation"]),
        evaluation=paired(candidate["_ics"]["evaluation"], base["_ics"]["evaluation"]),
    )


def origin_report(state, year, seed, controls, groups, references, ensemble_limit):
    records = completed(state, year, seed)
    reports = {}
    for name, record in records.items():
        report = model_report(state, record, year, seed)
        if report is not None:
            reports[name] = report
    for reference in references:
        for name, record in completed(reference, year, seed).items():
            if name not in reports:
                report = model_report(reference, record, year, seed)
                if report is not None:
                    reports[name] = report
    contrasts = [
        contrast(name, control, groups.get(name), reports[name], reports[control])
        for name, control in sorted(controls.items())
        if name in reports and control in reports
    ]
    ensembles = forward_selection(reports, ensemble_limit) if len(reports) > 1 else []
    for report in reports.values():
        # Los contrastes entre orígenes solo necesitan el IC por sesión: se liberan las filas.
        report["_splits"], report["_ranks"] = None, None
    return dict(
        year=year,
        completed=sorted(records),
        reports=reports,
        contrasts=contrasts,
        ensembles=ensembles,
    )


def aggregate(origins, controls, groups):
    """Contraste con las sesiones de todos los orígenes en los que están los dos modelos."""
    rows = []
    for name, control in sorted(controls.items()):
        merged = {kind: ({}, {}) for kind in ("decision", "evaluation")}
        years, means = [], []
        for origin in origins:
            reports = origin["reports"]
            if name not in reports or control not in reports:
                continue
            years.append(origin["year"])
            decision = (pooled(reports[name]["_ics"]), pooled(reports[control]["_ics"]))
            means.append(paired(*decision)["mean"])
            sides = dict(
                decision=decision,
                evaluation=(
                    reports[name]["_ics"]["evaluation"],
                    reports[control]["_ics"]["evaluation"],
                ),
            )
            for kind, (left, right) in sides.items():
                merged[kind][0].update({(origin["year"], *key): v for key, v in left.items()})
                merged[kind][1].update({(origin["year"], *key): v for key, v in right.items()})
        if not years:
            continue
        rows.append(
            dict(
                model=name,
                control=control,
                group=groups.get(name),
                origins=years,
                origin_means=means,
                mean_of_origins=float(np.mean([m for m in means if m is not None]))
                if means
                else None,
                decision=paired(*merged["decision"]),
                evaluation=paired(*merged["evaluation"]),
            )
        )
    return rows


def comparison(origins, other, seed):
    """Cada modelo de este estado frente al mismo modelo de otro, en sus sesiones comunes."""
    rows = []
    for origin in origins:
        records = completed(other, origin["year"], seed)
        for name in sorted(set(origin["reports"]) & set(records)):
            mine = origin["reports"][name]
            # Un modelo tomado del otro estado como referencia no se compara consigo mismo.
            if mine["state"] == str(other):
                continue
            report = model_report(other, records[name], origin["year"], seed)
            if report is None:
                continue
            rows.append(
                dict(
                    model=name,
                    year=origin["year"],
                    decision=paired(pooled(mine["_ics"]), pooled(report["_ics"])),
                    evaluation=paired(mine["_ics"]["evaluation"], report["_ics"]["evaluation"]),
                )
            )
    return rows


def build(state, years, seed, controls, groups, references=(), compare=None, ensemble_limit=6):
    origins = [
        origin_report(state, year, seed, controls, groups, references, ensemble_limit)
        for year in years
    ]
    document = dict(
        schema_version=1,
        kind="dgx_screening_report",
        state=str(state),
        references=[str(path) for path in references],
        seed=seed,
        min_assets=MIN_ASSETS,
        decision_splits=list(DECISION),
        origins=[
            dict(
                year=origin["year"],
                completed=origin["completed"],
                models={
                    name: {key: value for key, value in report.items() if not key.startswith("_")}
                    for name, report in origin["reports"].items()
                },
                contrasts=origin["contrasts"],
                ensembles=origin["ensembles"],
            )
            for origin in origins
        ],
        aggregate=aggregate(origins, controls, groups) if len(origins) > 1 else [],
        comparison=dict(state=str(compare), rows=comparison(origins, compare, seed))
        if compare
        else None,
    )
    return document


def _f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _contrast_rows(rows, lines):
    lines += [
        "| Variante | Control | Diferencia de decisión | t | Diferencia en evaluación | t |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in sorted(rows, key=lambda item: -(item["decision"]["mean"] or -9)):
        lines.append(
            f"| `{item['model']}` | `{item['control']}` | {_f(item['decision']['mean'])} | "
            f"{_f(item['decision']['t'], 2)} | {_f(item['evaluation']['mean'])} | "
            f"{_f(item['evaluation']['t'], 2)} |"
        )
    lines.append("")


def markdown(report):
    years = ", ".join(str(origin["year"]) for origin in report["origins"])
    lines = [
        f"# Criba de los orígenes {years}, semilla {report['seed']}",
        "",
        f"Estado: `{report['state']}`. IC de rangos por sesión. Se decide solo con validación y "
        "calibración; la evaluación es la evidencia fuera de muestra de cada origen.",
        "",
    ]
    if report["aggregate"]:
        lines += ["## Contrastes pareados con todos los orígenes", ""]
        _contrast_rows(report["aggregate"], lines)
    for origin in report["origins"]:
        lines += [
            f"## Origen {origin['year']}: {len(origin['models'])} modelos",
            "",
            "| Modelo | Nivel | Decisión | Validación | Calibración | Evaluación | "
            "ICIR de evaluación | Desviación por época | Horas |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        ordered = sorted(origin["models"].values(), key=lambda item: -(item["decision_ic"] or -9))
        for item in ordered:
            ic = item["ic"]
            cells = [
                _f(item["decision_ic"]),
                _f(ic["validation"]["mean"]),
                _f(ic["calibration"]["mean"]),
                _f(ic["evaluation"]["mean"]),
                _f(ic["evaluation"]["icir"], 2),
                _f(item["epoch_std"]),
                _f(item["hours"], 2),
            ]
            lines.append(f"| `{item['model']}` | {item['tier']} | " + " | ".join(cells) + " |")
        markets = sorted({market for item in ordered for market in item["ic_by_market"]})
        if markets:
            lines += ["", "IC de evaluación por mercado:", ""]
            lines.append("| Modelo | " + " | ".join(markets) + " |")
            lines.append("| --- | " + " | ".join("---" for _ in markets) + " |")
            for item in ordered:
                cells = [
                    _f((item["ic_by_market"].get(market) or {}).get("evaluation", {}).get("mean"))
                    for market in markets
                ]
                lines.append(f"| `{item['model']}` | " + " | ".join(cells) + " |")
        lines += ["", f"### Contrastes pareados del origen {origin['year']}", ""]
        _contrast_rows(origin["contrasts"], lines)
        lines += [
            f"### Combinaciones por media de rangos del origen {origin['year']}",
            "",
            "| Miembros | IC de decisión | IC de evaluación | ICIR de evaluación |",
            "| --- | --- | --- | --- |",
        ]
        for step in origin["ensembles"]:
            members = ", ".join(f"`{name}`" for name in step["members"])
            evaluation = step["evaluation"] or {}
            lines.append(
                f"| {members} | {_f(step['decision_ic'])} | {_f(evaluation.get('mean'))} | "
                f"{_f(evaluation.get('icir'), 2)} |"
            )
        lines.append("")
    if report["comparison"]:
        lines += [
            f"## Este estado frente a `{report['comparison']['state']}` en sus sesiones comunes",
            "",
            "| Modelo | Origen | Diferencia de decisión | t | Diferencia en evaluación | t |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for item in report["comparison"]["rows"]:
            lines.append(
                f"| `{item['model']}` | {item['year']} | {_f(item['decision']['mean'])} | "
                f"{_f(item['decision']['t'], 2)} | {_f(item['evaluation']['mean'])} | "
                f"{_f(item['evaluation']['t'], 2)} |"
            )
        lines.append("")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--year", type=int, action="append", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--controls", type=Path, help="JSON con controls {variante: control} y groups"
    )
    parser.add_argument("--reference", type=Path, action="append", default=[])
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    declared = json.loads(args.controls.read_text()) if args.controls else {}
    report = build(
        args.state,
        args.year,
        args.seed,
        declared.get("controls", {}),
        declared.get("groups", {}),
        references=args.reference,
        compare=args.compare,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1) + "\n"
    )
    (args.output / "report.md").write_text(markdown(report))
    count = sum(len(origin["models"]) for origin in report["origins"])
    print(f"{count} modelos en {len(report['origins'])} orígenes: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
