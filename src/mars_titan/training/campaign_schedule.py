"""Orden de la campaña ventana a ventana, de la base a la liberación de lo temporal.

Una ventana de campaña reúne las ventanas de todos los ámbitos con los cuatro tramos
idénticos, de modo que el ajuste conjunto y sus controles separados avanzan juntos. Su
nombre es el de la ventana de un ámbito presente en todas, el conjunto cuando existe.

Con `execution.order = "by_window"` la campaña recorre cada ventana en estas fases:

1. `base_search`: casos de búsqueda de todos los brazos con la semilla de búsqueda.
2. `selection`: elección del caso de cada brazo con la validación de esa ventana.
3. `selected_case_seeds`: el caso elegido con las demás semillas (y los traslados).
4. `online`: el control en línea, que parte del estado elegido de su padre en la ventana.
5. `adapters`: el posentrenamiento, que parte del estado elegido de la base en la ventana
   anterior y ajusta solo con las filas nuevas (`campaign_chain`).
6. `chain`: la selección del predictor de la cadena de cada brazo base y semilla.
7. `ablation` y `rl`: la ablación y la política, que lee la cadena de esta ventana y de
   todas las anteriores que necesita.
8. `comparison`: agregados por sesión de la comparación de esa ventana.
9. `release`: liberación de lo temporal de la ventana, tras sus agregados.

Después empieza la siguiente ventana. El plan solo admite dependencias hacia fases
anteriores de la misma ventana o hacia ventanas anteriores. La selección no es un trabajo,
sino la lectura del MAE de validación de los recibos de búsqueda, y las dos últimas fases
son los puntos de conexión de los agregados por ventana y de la retención rodante. Si la
campaña declara el walk-forward por etapas, el calendario crea las selecciones de la cadena
a partir de la etapa de adaptadores y exige a adaptadores y RL las dependencias del diseño.
"""

import argparse
import json
from pathlib import Path

from mars_titan.evaluation.splits import PARTITIONS

from . import campaign_chain

ORDERS = ("by_scope", "by_window")
PHASES = (
    "base_search",
    "selection",
    "selected_case_seeds",
    "online",
    "adapters",
    "chain",
    "ablation",
    "rl",
    "comparison",
    "release",
)
# Fase de cada clase (`stage`) de trabajo de la campaña base.
BASE_PHASES = dict(
    search="base_search",
    finalist="selected_case_seeds",
    carry="selected_case_seeds",
    online="online",
)
STAGES = ("adapters", "ablation", "rl")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def campaign_windows(campaign):
    """Ventanas de campaña en orden temporal, con la ventana de cada ámbito que las forma."""
    resolved = campaign["comparison_config"]["resolved_scopes"]
    groups = {}
    for scope in campaign["scopes"]:
        for window, fold in resolved[scope]["windows"].items():
            bounds = tuple(tuple(fold[name]) for name in PARTITIONS)
            groups.setdefault(bounds, {})[scope] = window
    ordered = sorted(groups.items(), key=lambda item: item[0][1][0])
    complete = [s for s in campaign["scopes"] if all(s in slot for _, slot in ordered)]
    _require(complete, "Ningún ámbito tiene ventana en todas las ventanas de la campaña")
    # Prefiere el ámbito con más de un mercado, el conjunto, para nombrar las ventanas.
    named = max(complete, key=lambda s: len(resolved[s]["markets"]))
    return [dict(id=slot[named], scopes=slot) for _, slot in ordered]


def window_pairs(campaign, window):
    """Pares (ámbito, ventana) que forman una ventana de campaña."""
    for row in campaign_windows(campaign):
        if row["id"] == window:
            return set(row["scopes"].items())
    raise ValueError(f"{window} no es una ventana de la campaña")


def window_jobs(campaign, jobs, window):
    """Trabajos de una ventana de campaña y las dependencias de ventanas anteriores.

    Las dependencias se conservan para que la reanudación las confirme antes de seguir.
    """
    pairs = window_pairs(campaign, window)
    by_id = {job["id"]: job for job in jobs}
    wanted = {job["id"] for job in jobs if (job["scope"], job["window"]) in pairs}
    pending = list(wanted)
    while pending:
        for dependency in by_id[pending.pop()].get("depends", ()):
            _require(dependency in by_id, f"La dependencia {dependency} no está en el plan")
            if dependency not in wanted:
                wanted.add(dependency)
                pending.append(dependency)
    return [job for job in jobs if job["id"] in wanted]


def stage_window(campaign, jobs, window):
    """Trabajos de una etapa posterior en una ventana y los pares que necesita de la base.

    Los pares incluyen la ventana ancla de cada trabajo, de la que parte un traslado.
    """
    jobs = window_jobs(campaign, jobs, window)
    pairs = {(job["scope"], name) for job in jobs for name in (job["window"], job["anchor"])}
    return jobs, pairs


def _position(campaign):
    return {
        pair: index
        for index, row in enumerate(campaign_windows(campaign))
        for pair in row["scopes"].items()
    }


def order_by_window(campaign, jobs):
    """Trabajos de la campaña base ventana a ventana, con la búsqueda antes que las semillas.

    El orden es estable, así que dentro de cada ventana y fase se conserva el del plan
    (ámbitos en el orden declarado y padres antes que sus brazos).
    """
    position = _position(campaign)
    ordered = sorted(
        jobs,
        key=lambda job: (
            position[job["scope"], job["window"]],
            PHASES.index(BASE_PHASES[job["stage"]]),
        ),
    )
    seen = set()
    for job in ordered:
        _require(
            all(dependency in seen for dependency in job["depends"]),
            f"{job['id']} queda antes que una de sus dependencias",
        )
        seen.add(job["id"])
    return ordered


def window_schedule(campaign, jobs, stages=None):
    """Fases de cada ventana de campaña con sus trabajos, comprobando las dependencias.

    `jobs` es el plan de la campaña base y `stages` asigna a `adapters`, `ablation` o `rl`
    el plan de esa etapa, que debe partir de la misma campaña. Con el walk-forward por
    etapas declarado, añade la fase `chain` y comprueba las dependencias del diseño.
    """
    stages = dict(stages or {})
    _require(set(stages) <= set(STAGES), "Las etapas posteriores son adapters, ablation o rl")
    if campaign.get("walk_forward_stages"):
        campaign_chain.check_staged(campaign, jobs, stages)
        if "adapters" in stages:
            stages["chain"] = campaign_chain.chain_jobs(campaign, jobs, stages["adapters"])
    rows = campaign_windows(campaign)
    position = _position(campaign)
    located = {}
    phases = [{phase: [] for phase in PHASES} for _ in rows]
    for name, stage_jobs in (("base", jobs), *stages.items()):
        for job in stage_jobs:
            phase = BASE_PHASES[job["stage"]] if name == "base" else name
            pair = job["scope"], job["window"]
            _require(pair in position, f"{job['id']} no pertenece a ninguna ventana")
            _require(job["id"] not in located, f"{job['id']} aparece dos veces")
            located[job["id"]] = position[pair], PHASES.index(phase)
            phases[position[pair]][phase].append(job["id"])
    for stage_jobs in (jobs, *stages.values()):
        for job in stage_jobs:
            for dependency in job.get("depends", ()):
                _require(
                    dependency in located and located[dependency] <= located[job["id"]],
                    f"{job['id']} depende de una fase posterior o ajena al plan",
                )
    schedule = []
    for index, row in enumerate(rows):
        decisions = sorted(
            {
                "/".join((job["scope"], job["window"], job["arm"]))
                for job in jobs
                if job["stage"] == "search" and position[job["scope"], job["window"]] == index
            }
        )
        entries = []
        for phase in PHASES:
            entry = dict(phase=phase, jobs=phases[index][phase])
            if phase == "selection":
                entry["decisions"] = decisions
            elif phase == "comparison":
                entry["scopes"] = row["scopes"]
            entries.append(entry)
        schedule.append(dict(window=row["id"], scopes=row["scopes"], phases=entries))
    return schedule


def main(argv=None):
    """Mostrar las fases de cada ventana y el número de trabajos, sin leer datos."""
    from mars_titan.posttraining import campaign_stage as adapter_stage
    from mars_titan.simulation import policy_plan

    from . import modality_ablation_stage
    from .campaign_plan import load_campaign, plan_campaign

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--adapter-stage", type=Path)
    parser.add_argument("--ablation-stage", type=Path)
    parser.add_argument("--rl-stage", type=Path)
    parser.add_argument("--output", type=Path, help="Escribir el orden completo con sus trabajos")
    args = parser.parse_args(argv)
    campaign = load_campaign(args.campaign)
    stages = {}
    for name, path, module in (
        ("adapters", args.adapter_stage, adapter_stage),
        ("ablation", args.ablation_stage, modality_ablation_stage),
        ("rl", args.rl_stage, policy_plan),
    ):
        if path is not None:
            stage = module.load_stage(path)
            _require(
                stage["campaign"]["sha256"] == campaign["sha256"],
                f"La etapa {name} parte de otra campaña",
            )
            stages[name] = module.plan_stage(stage)
    schedule = window_schedule(campaign, plan_campaign(campaign), stages)
    if args.output is not None:
        from mars_titan.data.storage import atomic_json

        atomic_json(
            args.output, dict(schema_version=1, campaign=campaign["sha256"], windows=schedule)
        )
    summary = [
        dict(
            window=row["window"],
            scopes=row["scopes"],
            jobs={entry["phase"]: len(entry["jobs"]) for entry in row["phases"] if entry["jobs"]},
            decisions=len(row["phases"][PHASES.index("selection")]["decisions"]),
        )
        for row in schedule
    ]
    print(json.dumps(dict(order=PHASES, windows=summary), ensure_ascii=False, indent=2))
    return 0
