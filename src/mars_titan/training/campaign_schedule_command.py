"""Orden `schedule` de la campaña: muestra las fases de cada ventana sin leer datos.

Carga la campaña y, si se indican, las etapas de adaptadores, ablación y políticas, y
muestra el orden de `campaign_schedule.window_schedule`. Vive aparte del cálculo del orden
porque necesita cargar las tres etapas, y la etapa de políticas, que usa
`campaign_schedule.stage_window`, no debe alcanzar el código de la etapa de adaptadores.
"""

import argparse
import json
from pathlib import Path

from .campaign_schedule import PHASES, _require, window_schedule


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
            if name == "adapters" and stage["design"] == module.STAGED:
                # Las selecciones de la cadena van con los adaptadores y el calendario las
                # pasa a su fase.
                stages[name] += module.plan_chain(stage, stages[name])
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
