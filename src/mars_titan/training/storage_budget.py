"""Estimar el disco de la campaña A, de su etapa de adaptadores y de su etapa de políticas.

La estimación no lee objetivos, muestras ni modelos. Toma el plan de cada etapa (base,
ablación de modalidades, adaptadores y políticas), los
recuentos reales de las vistas y los bytes medidos con tablas sintéticas por
`training.storage_measurements`, y suma por ámbito, ventana, familia y caso:

- tablas por fila con la disposición actual y con las alternativas sin pérdida,
- puntos de control con la rotación real y con la liberación al confirmar,
- índices de observaciones, informes, resúmenes por sesión y cachés temporales.

El pico recorre los trabajos en el orden del plan, porque la campaña ejecuta de uno en uno
los trabajos CUDA. Cada escenario de retención separa lo que se conserva al terminar de
lo que solo existe mientras corre un trabajo.
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.storage import atomic_json

from .campaign_plan import PLATEAU
from .campaign_storage import HELD_OUT, INDEXED, WRITERS, index_rows, job_footprint, load_storage

# Escenarios de retención de las predicciones del ajuste base. `rows` decide qué tablas se
# guardan por fila, `layout` su disposición y `compacted` si la disposición se aplica
# después de escribir, de modo que el original existe mientras corre el trabajo.
SCENARIOS = {
    "current_without_release": dict(layout="current", rows="all", compacted=False, release=False),
    "current": dict(layout="current", rows="all", compacted=False, release=True),
    "large_groups": dict(layout="large_groups", rows="all", compacted=False, release=True),
    "shared_rows": dict(layout="shared_rows", rows="all", compacted=True, release=True),
    "needed_large_groups": dict(layout="large_groups", rows="needed", compacted=True, release=True),
    "needed_shared_rows": dict(layout="shared_rows", rows="needed", compacted=True, release=True),
}
QUANTILE_HEAD = "quantile_head_v1"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def campaign_jobs(campaign_path, extensions_path=None):
    """Campaña cargada, con las secciones preparadas de su variante si se dan, y sus trabajos."""
    from .campaign_plan import ONLINE, load_campaign, plan_campaign

    campaign = load_campaign(campaign_path)
    if extensions_path is not None:
        from .campaign_extensions import extended_campaign, load_extensions

        campaign = extended_campaign(load_extensions(extensions_path), campaign)
    # El control en línea no tiene todavía ejecutor ni informe que medir. Su disco se
    # contará con el informe de su ejecutor.
    return campaign, [job for job in plan_campaign(campaign) if job["kind"] != ONLINE]


def view_reports(views):
    """Recuentos y manifiesto de cada ventana desde `report.json`, sin abrir datos."""
    result = {}
    for scope, directory in views.items():
        report, _ = read_manifest(Path(directory) / "report.json", 1024**2)
        result[scope] = {}
        for row in report["folds"]:
            manifest = Path(directory) / row["id"] / "manifest.json"
            assets, _ = read_manifest(manifest, 64 * 1024**2)
            result[scope][row["id"]] = dict(
                counts=dict(row["counts"], flows=len(assets["assets"])),
                manifest=str(manifest),
            )
    return result


def measure_windows(reports, scratch, cache):
    """Medir cada ventana una vez y conservar las medidas en `cache` para repetir el cálculo."""
    from .storage_measurements import measure_view

    cache = Path(cache)
    measured = json.loads(cache.read_text()) if cache.is_file() else {}
    for scope, windows in reports.items():
        for window, record in windows.items():
            key = f"{scope}/{window}"
            if key not in measured:
                measured[key] = measure_view(record["manifest"], Path(scratch) / "windows")
                atomic_json(cache, measured)
    return measured


def row_bytes(measured):
    """Bytes por fila máximos de cada escritor, disposición y tramo entre las ventanas medidas.

    Son los coeficientes de la fórmula: con ellos y las filas reservadas de cada ventana,
    `formula_windows` da la medida de una ventana sin volver a escribir tablas.
    """
    writers, shared = defaultdict(dict), {}
    for window in measured.values():
        for partition, record in window.items():
            rows = record["rows"]
            for writer, sizes in record["writers"].items():
                for layout, value in sizes.items():
                    slot = writers[writer].setdefault(layout, {})
                    slot[partition] = max(slot.get(partition, 0.0), value / rows)
            shared[partition] = max(shared.get(partition, 0.0), record["shared_rows_bytes"] / rows)
    return dict(writers=dict(writers), shared_rows=shared, windows=len(measured))


def formula_windows(counts, coefficients):
    """Medida de cada ventana como filas reservadas por bytes por fila, redondeada hacia arriba."""
    measured = {}
    for scope, windows in counts.items():
        for window, value in windows.items():
            measured[f"{scope}/{window}"] = {
                partition: dict(
                    rows=value[partition],
                    shared_rows_bytes=math.ceil(
                        value[partition] * coefficients["shared_rows"][partition]
                    ),
                    writers={
                        writer: {
                            layout: math.ceil(value[partition] * per_row[partition])
                            for layout, per_row in layouts.items()
                        }
                        for writer, layouts in coefficients["writers"].items()
                    },
                )
                for partition in HELD_OUT
            }
    return measured


def selected_jobs(jobs):
    """Un ajuste elegido por ámbito, ventana, brazo y semilla, sin conocer resultados.

    Un finalista o un traslado es el único trabajo de su semilla. Entre las búsquedas,
    cualquiera puede ganar y todas tienen las mismas filas, así que se toma la primera. La
    meseta de un ajuste conjunto no escribe tablas y nunca es la elegida: lo es su
    continuación, que va después en el plan.
    """
    chosen = {}
    for job in jobs:
        if job.get("phase") == PLATEAU:
            continue
        chosen.setdefault((job["scope"], job["window"], job["arm"], job["seed"]), job["id"])
    return set(chosen.values())


def needed_partitions(job, campaign, selected):
    """Tramos cuyas filas lee algún consumidor: evaluación del elegido y, con cuantiles,
    su calibración. Los auxiliares y los no elegidos solo necesitan agregados."""
    arms = campaign["comparison_config"]["arms"]
    if job["id"] not in selected or job["arm"] not in arms:
        return ()
    quantile = arms[job["arm"]]["output"] == QUANTILE_HEAD
    return ("calibration", "evaluation") if quantile else ("evaluation",)


def _family(job):
    return job["family"] if job["model"] != "cm_v1_core" else "cm_v1_core"


def _search(job):
    return (job["scope"], job["window"], job["arm"], job["seed"])


def base_estimate(campaign, jobs, reports, measured, storage, extras):
    """Bytes por escenario, por grupo y en el pico del recorrido del plan.

    Cuando solo se guardan por fila los tramos del elegido, el ganador de una búsqueda no
    se conoce hasta confirmar todos sus casos, así que las filas de los casos confirmados
    de una búsqueda abierta cuentan en el pico hasta su último caso. La meseta de un ajuste
    conjunto solo suma sus estados e índices: sus tablas y agregados son los de la
    continuación.
    """
    selected = selected_jobs(jobs)
    aggregate = extras["aggregate_bytes_per_row"]
    last = {
        _search(job): i
        for i, job in enumerate(jobs)
        if job["stage"] == "search" and job.get("phase") != PLATEAU
    }
    groups = defaultdict(lambda: defaultdict(int))
    totals = {}
    for name, scenario in SCENARIOS.items():
        cumulative, peak, worst, shared_seen = 0, 0, None, set()
        parts, provisional = defaultdict(int), defaultdict(int)
        for index, job in enumerate(jobs):
            window = measured[f"{job['scope']}/{job['window']}"]
            counts = reports[job["scope"]][job["window"]]["counts"]
            writer = WRITERS[job["model"]]

            def written(_, partition, window=window, writer=writer):
                return window[partition]["writers"][writer]["current"]

            footprint = job_footprint(
                job, counts, storage, release=scenario["release"], prediction_bytes=written
            )
            original = footprint["retained"]["predictions"]
            plateau = job.get("phase") == PLATEAU
            needed = (
                HELD_OUT
                if scenario["rows"] == "all"
                else needed_partitions(job, campaign, selected)
            )
            kept = 0
            for partition in () if plateau else HELD_OUT:
                if partition in needed:
                    kept += window[partition]["writers"][writer][scenario["layout"]]
                else:
                    kept += int(counts[partition] * aggregate)
                if (
                    scenario["layout"] == "shared_rows"
                    and partition in needed
                    and (job["scope"], job["window"], partition) not in shared_seen
                ):
                    shared_seen.add((job["scope"], job["window"], partition))
                    kept += window[partition]["shared_rows_bytes"]
            retained = footprint["retained_bytes"] - original + kept
            transient = footprint["transient_bytes"] + (original if scenario["compacted"] else 0)
            moment = cumulative + retained + transient + sum(provisional.values())
            if moment > peak:
                peak, worst = moment, job["id"]
            cumulative += retained
            if scenario["rows"] == "needed" and job["stage"] == "search" and not plateau:
                key, winner = _search(job), needed_partitions(job, campaign, {job["id"]})
                for partition in set(winner) - set(needed):
                    provisional[key] += window[partition]["writers"][writer][scenario["layout"]]
                    provisional[key] -= int(counts[partition] * aggregate)
                if last[key] == index:
                    provisional.pop(key, None)
            for key, value in footprint["retained"].items():
                parts[key] += value if key != "predictions" else 0
            parts["predictions"] += kept
            group = (job["scope"], job["window"], _family(job), job["stage"])
            groups[group][name] += retained
            if name == "current":
                groups[group]["jobs"] += 1
                groups[group]["held_out_rows"] += 0 if plateau else sum(counts[p] for p in HELD_OUT)
        totals[name] = dict(
            retained_bytes=cumulative, peak_bytes=peak, peak_job=worst, retained_parts=dict(parts)
        )
    rows = [
        dict(scope=s, window=w, family=f, stage=t, **values)
        for (s, w, f, t), values in sorted(groups.items())
    ]
    return dict(totals=totals, groups=rows, selected_jobs=len(selected), jobs=len(jobs))


ABLATION_LAYOUTS = ("current", "large_groups", "shared_rows")


def ablation_estimate(path, reports, measured, storage):
    """Etapa de ablación de modalidades: una evaluación por trabajo con el escritor del modelo.

    Las filas y objetivos son los de la evaluación base, así que con la tabla común solo se
    añaden los decimales de cada trabajo. Los traslados cronológicos escriben además el
    índice de su evaluación con el calentamiento, que la etapa no libera al confirmar.
    """
    from .modality_ablation_stage import load_stage, plan_stage

    stage = load_stage(path)
    jobs = plan_stage(stage)
    index = storage["index"]
    tables = dict.fromkeys(ABLATION_LAYOUTS, 0)
    indices, writing, rows = 0, 0, 0
    models = defaultdict(int)
    for job in jobs:
        counts = reports[job["scope"]][job["window"]]["counts"]
        written = measured[f"{job['scope']}/{job['window']}"]["evaluation"]
        sizes = written["writers"][WRITERS[job["model"]]]
        for layout in ABLATION_LAYOUTS:
            tables[layout] += sizes[layout]
        rows += counts["evaluation"]
        models[job["model"]] += 1
        build = 0
        if job["model"] in INDEXED:
            events = index_rows(
                counts, index["warmup_partition"], partitions=("evaluation",), train=False
            )
            indices += int(events * index["bytes_per_row"])
            build = int(2 * counts["evaluation"] * index["build_bytes_per_row"])
        writing = max(writing, sizes["current"] + build)
    reports_bytes = len(jobs) * storage["job_report_bytes"]
    layouts = {}
    for layout, value in tables.items():
        retained = value + reports_bytes + indices
        layouts[layout] = dict(
            retained_bytes=retained,
            retained_without_indices=retained - indices,
            peak_bytes=retained + writing,
        )
    return dict(
        path=stage["path"],
        jobs=len(jobs),
        models=dict(models),
        evaluation_rows=rows,
        index_bytes=indices,
        layouts=layouts,
    )


# Retención de las tablas de la etapa de adaptadores. Con la tabla común de la campaña base,
# cada ajuste guarda sus cinco cuantiles en float64 y su padre en float32. `needed` guarda por
# fila la calibración y la evaluación, que lee la comparación, y agrega la validación.
ADAPTER_SCENARIOS = {
    "current": dict(layout="current", rows=HELD_OUT),
    "large_groups": dict(layout="large_groups", rows=HELD_OUT),
    "shared_rows": dict(layout="shared_rows", rows=HELD_OUT),
    "needed_shared_rows": dict(layout="shared_rows", rows=("calibration", "evaluation")),
}


def adapter_estimate(path, reports, extras):
    """Etapa de adaptadores: tablas, estados, cachés de padres y corpus ordenado por ventana.

    Con la lectura por bloques de la vista no hay corpus ordenado. Con él, la copia de
    una ventana se libera tras sus ajustes y, mientras se prepara, coexisten la entrada
    sin ordenar y el archivo ordenado de entrenamiento. Una disposición
    distinta de la actual se aplica al confirmar, así que la tabla original de un ajuste
    existe mientras corre.
    """
    from mars_titan.posttraining.campaign_stage import load_stage, plan_stage

    stage = load_stage(path)
    jobs = plan_stage(stage)
    adapter = extras["adapter"]
    aggregate = extras["aggregate_bytes_per_row"]
    fixed = adapter["state_bytes"] * adapter["retained_states"] + adapter["job_report_bytes"]
    parents = {(j["scope"], j["window"], j["base_arm"], j["seed"]) for j in jobs}
    windows = []
    for job in jobs:
        key = (job["scope"], job["window"])
        if not windows or windows[-1][0] != key:
            windows.append((key, []))
        windows[-1][1].append(job)
    result = {}
    for name, scenario in ADAPTER_SCENARIOS.items():
        per_row = adapter[f"row_bytes_{scenario['layout']}"]
        compacted = scenario["layout"] == "shared_rows"
        cumulative, peak, worst = 0, 0, None
        for (scope, window), members in windows:
            counts = reports[scope][window]["counts"]
            ordered = preparing = 0
            if stage["cohort_reading"]["source"] == "ordered_corpus":
                rows = counts["train"] + counts["validation"]
                ordered = rows * adapter["ordered_row_bytes"]
                preparing = counts["train"] * adapter["input_row_bytes"]
            kept = sum(
                counts[p] * (per_row if p in scenario["rows"] else aggregate) for p in HELD_OUT
            )
            outputs = len(members) * (int(kept) + fixed)
            original = int(sum(counts[p] for p in HELD_OUT) * adapter["row_bytes_current"])
            caches = sum(1 for p in parents if p[:2] == (scope, window))
            caches *= adapter["parent_cache_bytes"]
            during = ordered + outputs + caches + (original if compacted else 0)
            moment = cumulative + max(ordered + preparing, during)
            if moment > peak:
                peak, worst = moment, f"{scope}/{window}"
            cumulative += outputs + caches
        result[name] = dict(retained_bytes=cumulative, peak_bytes=peak, peak_window=worst)
    return dict(path=stage["path"], jobs=len(jobs), parents=len(parents), layouts=result)


def rl_estimate(path, campaign, extras, extensions_path=None):
    """Etapa de políticas: cintas por ámbito, mercado, ancla y predictor.

    Los ejecutores de ajuste de PPO y KLPO no existen todavía, así que sus artefactos no
    se pueden estimar y quedan declarados como pendientes.
    """
    from mars_titan.simulation import policy_plan

    stage = policy_plan.load_stage(path)
    if extensions_path is not None:
        from .campaign_extensions import extended_policies, load_extensions

        stage = extended_policies(load_extensions(extensions_path), stage, campaign)
    jobs = policy_plan.plan_stage(stage)
    anchors = {}
    for job in jobs:
        key = (job["scope"], job["market"], job["anchor"], job["predictor"])
        anchors[key] = len(job["train"]) + 2
    tapes = sum(anchors.values())
    counts = policy_plan.count_stage(stage, jobs)
    return dict(
        path=stage["path"],
        jobs={k: v for k, v in counts.items() if k.endswith("_jobs")},
        tapes=tapes,
        tape_bytes=extras["tape_bytes"],
        retained_bytes=tapes * extras["tape_bytes"],
        pending="Los ejecutores de ajuste de PPO y KLPO no existen, sin artefactos estimables",
    )


def estimate(args):
    """Estimación de todas las etapas declaradas, con vistas medidas o con la fórmula.

    Con `--views` se miden las ventanas que falten en `--cache`. Con `--counts` y
    `--row-bytes` cada ventana sale de sus filas reservadas por los bytes por fila medidos,
    para recalcular en cuanto cambian los recuentos.
    """
    storage = load_storage(args.storage)
    campaign, jobs = campaign_jobs(args.campaign, args.extensions)
    if args.counts is not None:
        _require(args.row_bytes is not None, "La fórmula necesita los bytes por fila medidos")
        counts = json.loads(Path(args.counts).read_text())
        reports = {s: {w: dict(counts=c) for w, c in v.items()} for s, v in counts.items()}
        coefficients = json.loads(Path(args.row_bytes).read_text())
        measured = formula_windows(counts, coefficients)
    else:
        _require(args.views and args.cache and args.scratch, "Faltan vistas, caché o scratch")
        views = {}
        for value in args.views:
            scope, _, directory = value.partition("=")
            views[scope] = Path(directory)
        reports = view_reports(views)
        measured = measure_windows(reports, args.scratch, args.cache)
        coefficients = row_bytes(measured)
    _require(set(reports) == set(campaign["scopes"]), "Faltan recuentos de algún ámbito")
    extras = json.loads(Path(args.extras).read_text())
    result = dict(
        schema_version=1,
        kind="historical_masked_campaign_storage_estimate",
        storage=dict(path=storage["path"], sha256=storage["sha256"]),
        scenarios=SCENARIOS,
        row_bytes=coefficients,
        base=base_estimate(campaign, jobs, reports, measured, storage, extras),
        counts={s: {w: r["counts"] for w, r in v.items()} for s, v in reports.items()},
        final_test_opened=False,
    )
    if args.ablation_stage is not None:
        result["ablation"] = ablation_estimate(args.ablation_stage, reports, measured, storage)
    if args.adapter_stage is not None:
        result["adapters"] = adapter_estimate(args.adapter_stage, reports, extras)
    if args.rl_stage is not None:
        result["policies"] = rl_estimate(args.rl_stage, campaign, extras, args.extensions)
    if args.schedule is not None:
        from .rolling_retention import load_schedule
        from .rolling_storage import rolling_estimate, rolling_inputs, window_increment

        windows = load_schedule(args.schedule, campaign)
        stages = rolling_inputs(
            jobs,
            windows,
            extras,
            ablation=args.ablation_stage,
            adapters=args.adapter_stage,
            rl=args.rl_stage,
        )
        rolling = rolling_estimate(
            jobs,
            windows,
            result["counts"],
            measured,
            storage,
            extras,
            ordered_copy=not args.adapter_blocks,
            **stages,
        )
        result["rolling"] = dict(
            rolling,
            adapter_corpus="blocks" if args.adapter_blocks else "ordered_copy",
            increment={scenario: window_increment(rolling, scenario) for scenario in rolling},
        )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--extensions", type=Path)
    parser.add_argument("--storage", type=Path, required=True)
    parser.add_argument("--ablation-stage", type=Path)
    parser.add_argument("--adapter-stage", type=Path)
    parser.add_argument("--rl-stage", type=Path)
    parser.add_argument("--schedule", type=Path, help="Orden por ventanas de la retención v2")
    parser.add_argument(
        "--adapter-blocks",
        action="store_true",
        help="Los adaptadores leen la vista por bloques, sin copia ordenada",
    )
    parser.add_argument("--extras", type=Path, required=True)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--views", action="append")
    sources.add_argument("--counts", type=Path)
    parser.add_argument("--row-bytes", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--scratch", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = estimate(args)
    atomic_json(args.output, result)
    print(json.dumps(result["base"]["totals"], indent=2))
    if "rolling" in result:
        print(
            json.dumps(
                {
                    scenario: {
                        k: result["rolling"][scenario][k]
                        for k in ("retained_bytes", "peak_bytes", "peak_at")
                    }
                    for scenario in ("all_regenerated", "none_regenerated")
                },
                indent=2,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
