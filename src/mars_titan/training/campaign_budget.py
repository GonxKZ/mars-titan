"""Proyectar las horas de una campaña con un caudal supuesto o medido, sin ajustar nada.

La proyección aplica la fórmula de `campaign_throughput.estimate_hours` (filas de ajuste de
cada ventana por épocas entre caudal, más las pasadas de validación, las predicciones
finales y el calentamiento de las familias con memoria) a los trabajos exactos del plan.
El caudal puede ser uno común en muestras-época por segundo, con la inferencia a un múltiplo
declarado, o los caudales por familia y caso de un informe de `throughput`. Las épocas
pueden sustituirse por las efectivas previstas de una parada temprana.

Las filas de cada ámbito y ventana salen de las vistas preparadas o, antes de prepararlas,
de los objetivos residuales con la misma regla que las vistas: una fila cuenta en un tramo
si tiene objetivo, su decisión cae en el tramo y su etiqueta madura antes de su final.

Con el walk-forward por etapas (`campaign_chain`), cada ventana con padre ajusta los
adaptadores solo con sus filas nuevas, que se cuentan con la misma regla, y el padre
congelado de la ventana anterior predice validación, calibración y evaluación.

Tabulares y políticas no escalan con el caudal neuronal. Entran como horas fijas
declaradas, y el factor de caudal necesario para un objetivo de horas es
horas neuronales / (objetivo - horas fijas). Este módulo no reserva la GPU ni lee vistas
salvo que se le pidan sus recuentos.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest
from mars_titan.evaluation.splits import PARTITIONS, build_folds

from . import campaign_chain
from .campaign_plan import NEURAL, _arm_specs, _require, load_campaign
from .campaign_throughput import (
    ABLATION_STAGE,
    CHRONOLOGICAL,
    POSTTRAINING,
    _hours,
    _slowest,
    estimate_hours,
    validated_job_seconds,
)

COUNTS_KIND = "masked_campaign_window_counts"
BUDGET_KIND = "masked_campaign_budget_projection"


def _micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def target_window_counts(labels_root, protocol):
    """Filas por ventana y tramo de un mercado a partir de sus objetivos residuales.

    Lee de cada activo solo la decisión, la maduración y si hay objetivo. Es la misma
    regla de purga por intervalo de etiqueta que aplica la preparación de las vistas v2.
    """
    folds = build_folds(protocol)
    bounds = [[(_micros(f[n][0]), _micros(f[n][1])) for n in PARTITIONS] for f in folds]
    totals = np.zeros((len(folds), len(PARTITIONS)), dtype=np.int64)
    paths = sorted((Path(labels_root) / protocol["market"]).glob("*/labels.parquet"))
    _require(paths, f"No hay objetivos de {protocol['market']} en {labels_root}")
    for path in paths:
        table = pq.read_table(path, columns=["prediction_at", "target_available_at", "target"])
        valid = table["target"].is_valid().to_numpy(zero_copy_only=False)
        moment = table["prediction_at"].cast(pa.int64()).to_numpy()[valid]
        maturity = table["target_available_at"].cast(pa.int64()).to_numpy()[valid]
        for i, fold in enumerate(bounds):
            for j, (start, end) in enumerate(fold):
                inside = (moment >= start) & (moment < end) & (maturity < end)
                totals[i, j] += int(np.count_nonzero(inside))
    return {
        fold["id"]: dict(zip(PARTITIONS, map(int, totals[i]), strict=True))
        for i, fold in enumerate(folds)
    }


def target_posttraining_rows(labels_root, protocol):
    """Filas nuevas del posentrenamiento de cada ventana con padre, desde los objetivos.

    Cuentan las filas con objetivo, decisión en `campaign_chain.posttraining_rows` y etiqueta
    madura antes del final del tramo de ajuste, como el tramo `train` de la vista.
    """
    folds = build_folds(protocol)
    spans = [
        (fold["id"], *map(_micros, campaign_chain.posttraining_rows(parent, fold)))
        for parent, fold in zip(folds, folds[1:], strict=False)
    ]
    totals = dict.fromkeys((window for window, _, _ in spans), 0)
    for path in sorted((Path(labels_root) / protocol["market"]).glob("*/labels.parquet")):
        table = pq.read_table(path, columns=["prediction_at", "target_available_at", "target"])
        valid = table["target"].is_valid().to_numpy(zero_copy_only=False)
        moment = table["prediction_at"].cast(pa.int64()).to_numpy()[valid]
        maturity = table["target_available_at"].cast(pa.int64()).to_numpy()[valid]
        for window, start, end in spans:
            inside = (moment >= start) & (moment < end) & (maturity < end)
            totals[window] += int(np.count_nonzero(inside))
    return totals


def campaign_posttraining_rows(campaign, labels_root):
    """Filas nuevas por ámbito y ventana con padre, sumando los mercados de cada ámbito."""
    resolved = campaign["comparison_config"]["resolved_scopes"]
    rows = {}
    for scope in campaign["scopes"]:
        by_market = [
            target_posttraining_rows(labels_root, protocol)
            for protocol in resolved[scope]["protocols"].values()
        ]
        rows[scope] = {window: sum(part[window] for part in by_market) for window in by_market[0]}
    return rows


def read_posttraining_rows(path, campaign):
    """Filas nuevas declaradas en un informe de recuentos, para cada ventana con padre."""
    document, _ = read_manifest(Path(path), 16 * 1024**2)
    rows = document.get("posttraining_rows")
    _require(
        isinstance(rows, dict)
        and set(campaign["scopes"]) <= set(rows)
        and all(
            list(rows[scope]) == [w for w, _ in campaign_chain.scope_windows(campaign, scope)][1:]
            and all(type(v) is int and v > 0 for v in rows[scope].values())
            for scope in campaign["scopes"]
        ),
        "El informe no declara las filas nuevas de cada ventana con padre",
    )
    return {scope: rows[scope] for scope in campaign["scopes"]}


def staged_posttraining_hours(stage, counts, fresh, rates):
    """Horas de la etapa de adaptadores en el walk-forward por etapas.

    Solo las ventanas con padre tienen trabajos. Cada caso de la matriz recorre sus filas
    nuevas en lugar del tramo de ajuste completo y valida, calibra y evalúa como en la base.
    Cada padre guarda una vez la caché de filas nuevas y validación y, congelado, predice
    validación, calibración y evaluación, con la inferencia neuronal más lenta de su brazo.
    """
    from mars_titan.posttraining.campaign_stage import plan_stage

    measured, neural = rates[POSTTRAINING], rates[NEURAL]
    epochs = stage["matrix"]["budget"]["epochs"]
    jobs = [job for job in plan_stage(stage) if job["window"] in fresh[job["scope"]]]
    parents = {}
    for job in jobs:
        parents.setdefault((job["scope"], job["window"], job["base_arm"], job["seed"]), job)
    first = {job["id"] for job in parents.values()}

    def seconds(job):
        rows = dict(counts[job["scope"]][job["window"]], train=fresh[job["scope"]][job["window"]])
        rate = _slowest([points[job["point"]] for points in measured[job["base_arm"]].values()])
        total = validated_job_seconds(job, rows, rate, epochs)
        if job["id"] in first:
            cache = _slowest(neural[job["base_arm"]].values())["inference"]
            held = rows["validation"] + rows["calibration"] + rows["evaluation"]
            total += (rows["train"] + rows["validation"] + held) / cache
        return total

    return dict(_hours(jobs, seconds), frozen_parent_predictions=len(first))


def campaign_window_counts(campaign, labels_root):
    """Filas por ámbito y ventana de la campaña, sumando los mercados de cada ámbito."""
    resolved = campaign["comparison_config"]["resolved_scopes"]
    counts = {}
    for scope in campaign["scopes"]:
        by_market = [
            target_window_counts(labels_root, protocol)
            for protocol in resolved[scope]["protocols"].values()
        ]
        counts[scope] = {
            window: {name: sum(rows[window][name] for rows in by_market) for name in PARTITIONS}
            for window in resolved[scope]["windows"]
        }
    return counts


def read_counts(path, campaign):
    """Recuentos declarados en un informe, que deben cubrir los ámbitos y ventanas del plan."""
    document, digest = read_manifest(Path(path), 16 * 1024**2)
    resolved = campaign["comparison_config"]["resolved_scopes"]
    _require(
        isinstance(document, dict)
        and document.get("kind") == COUNTS_KIND
        and isinstance(document.get("counts"), dict)
        and set(campaign["scopes"]) <= set(document["counts"])
        and all(
            list(document["counts"][scope]) == list(resolved[scope]["windows"])
            and all(
                set(row) == set(PARTITIONS) and all(type(v) is int and v >= 0 for v in row.values())
                for row in document["counts"][scope].values()
            )
            for scope in campaign["scopes"]
        ),
        "Los recuentos no cubren exactamente los ámbitos y ventanas de la campaña",
    )
    return {scope: document["counts"][scope] for scope in campaign["scopes"]}, digest


def uniform_rates(campaign, rate, *, inference_ratio=3.0, stage=None):
    """Caudales iguales para todos los casos, en la forma de un informe de `throughput`.

    `rate` son muestras-época de ajuste por segundo y la inferencia corre a
    `inference_ratio` veces ese caudal. Con `stage`, cubre también sus puntos de adaptación.
    """
    _require(
        all(
            isinstance(v, (int, float)) and math.isfinite(v) and v > 0
            for v in (rate, inference_ratio)
        ),
        "El caudal y el múltiplo de inferencia deben ser positivos",
    )
    flat = dict(train=float(rate), inference=float(rate * inference_ratio))
    rates = {}
    for spec in _arm_specs(campaign):
        if spec["family"] == NEURAL:
            rates.setdefault(NEURAL, {})[spec["arm"]] = {
                name: dict(flat) for name, _ in spec["candidates"]
            }
        elif spec["family"] in CHRONOLOGICAL:
            rates.setdefault(spec["family"], {})[spec["arm"]] = dict(
                options=dict(recipe=dict(train=flat["train"], peak_vram_allocated_bytes=0)),
                declared_option="recipe",
                inference=flat["inference"],
            )
    if stage is not None:
        from mars_titan.posttraining.campaign_stage import plan_stage

        points = {}
        for job in plan_stage(stage):
            points.setdefault(job["base_arm"], set()).add(job["point"])
        rates[POSTTRAINING] = {
            arm: {"uniform": {point: dict(flat) for point in sorted(names)}}
            for arm, names in points.items()
        }
    return rates


def project(
    campaign,
    counts,
    rates,
    *,
    epochs=None,
    stage=None,
    ablation_stage=None,
    fixed_hours=None,
    target_hours=None,
    fresh=None,
):
    """Horas por etapa y factor de caudal necesario para un objetivo de horas.

    `fixed_hours` son las horas que no escalan con el caudal neuronal (tabulares y
    políticas), declaradas aparte porque no se miden sin ajustar. El factor solo se calcula
    con las dos cifras y un objetivo mayor que las horas fijas. Con `fresh`, las filas nuevas
    de cada ventana con padre, los adaptadores siguen el walk-forward por etapas.
    """
    estimate = estimate_hours(
        campaign, counts, rates, stage=stage, ablation_stage=ablation_stage, epochs=epochs
    )
    families = {
        name: family.get("hours", (family.get("options") or {}).get("recipe", {}).get("hours"))
        for name, family in estimate["families"].items()
    }
    base = math.fsum(v for k, v in families.items() if k != POSTTRAINING and v is not None)
    adapters = families.get(POSTTRAINING)
    if fresh is not None and stage is not None and POSTTRAINING in rates and NEURAL in rates:
        adapters = staged_posttraining_hours(stage, counts, fresh, rates)["hours"]
        families[POSTTRAINING] = adapters
    ablation = (estimate.get(ABLATION_STAGE) or {}).get("hours")
    stages = dict(base=base, adapters=adapters, ablation=ablation)
    neural = math.fsum(v for v in stages.values() if v is not None)
    factor = None
    if fixed_hours is not None and target_hours is not None:
        _require(target_hours > fixed_hours >= 0, "El objetivo debe superar las horas fijas")
        factor = neural / (target_hours - fixed_hours)
    return dict(
        schema_version=1,
        kind=BUDGET_KIND,
        campaign=dict(name=campaign["name"], sha256=campaign["sha256"]),
        epochs=campaign["rule"]["max_epochs"] if epochs is None else epochs,
        adapters_design="staged_chain_v1" if fresh is not None else "same_window_parent",
        stages=stages,
        families=families,
        neural_hours=neural,
        fixed_hours=fixed_hours,
        target_hours=target_hours,
        total_hours=None if fixed_hours is None else neural + fixed_hours,
        throughput_factor_needed=factor,
        without_estimate=estimate["total_gpu_hours"]["without_estimate"],
        estimate=estimate,
        assumptions=[
            *estimate["assumptions"],
            "Las épocas indicadas sustituyen a las de la regla de parada en todos los ajustes "
            "neuronales de la campaña base, no en los adaptadores",
            "Tabulares y políticas entran como horas fijas declaradas, sin medir",
        ],
        optimizer_steps=0,
        scientific_training_started=False,
    )


def main(argv=None):
    from mars_titan.posttraining.campaign_stage import load_stage

    from . import modality_ablation_stage
    from .masked_campaign import _views_argument, scope_views

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    rows = parser.add_mutually_exclusive_group(required=True)
    rows.add_argument("--counts", type=Path, help="Informe de recuentos por ámbito y ventana")
    rows.add_argument("--targets", type=Path, help="Carpeta labels de los objetivos residuales")
    rows.add_argument("--views", action="append", help="Vistas preparadas ÁMBITO=DIRECTORIO")
    speed = parser.add_mutually_exclusive_group(required=True)
    speed.add_argument("--rate", type=float, help="Muestras-época de ajuste por segundo")
    speed.add_argument("--throughput", type=Path, help="Informe medido de `throughput`")
    parser.add_argument("--inference-ratio", type=float, default=3.0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--stage", type=Path)
    parser.add_argument("--ablation-stage", type=Path)
    parser.add_argument("--fixed-hours", type=float)
    parser.add_argument("--target-hours", type=float)
    parser.add_argument("--write-counts", type=Path)
    args = parser.parse_args(argv)
    campaign = load_campaign(args.campaign)
    staged = campaign.get("walk_forward_stages") is not None
    fresh = None
    if args.counts is not None:
        counts, _ = read_counts(args.counts, campaign)
        if staged:
            fresh = read_posttraining_rows(args.counts, campaign)
    elif args.targets is not None:
        counts = campaign_window_counts(campaign, args.targets)
        if staged:
            fresh = campaign_posttraining_rows(campaign, args.targets)
    else:
        views = _views_argument(args.views)
        counts = {
            scope: {
                w: v["counts"]
                for w, v in scope_views(views[scope], scope, campaign)["windows"].items()
            }
            for scope in campaign["scopes"]
        }
    if args.write_counts is not None:
        from mars_titan.data.storage import atomic_json

        document = dict(schema_version=1, kind=COUNTS_KIND, counts=counts)
        if fresh is not None:
            document["posttraining_rows"] = fresh
        atomic_json(args.write_counts, document)
    stage = None if args.stage is None else load_stage(args.stage)
    ablation = None
    if args.ablation_stage is not None:
        ablation = modality_ablation_stage.load_stage(args.ablation_stage)
    if args.rate is not None:
        rates = uniform_rates(
            campaign, args.rate, inference_ratio=args.inference_ratio, stage=stage
        )
    else:
        rates = read_manifest(args.throughput, 64 * 1024**2)[0]["rates"]
    report = project(
        campaign,
        counts,
        rates,
        epochs=args.epochs,
        stage=stage,
        ablation_stage=ablation,
        fixed_hours=args.fixed_hours,
        target_hours=args.target_hours,
        fresh=fresh,
    )
    report.pop("estimate")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
