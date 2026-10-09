"""Orden única de la campaña con máscaras: comprobar, preparar, ejecutar, publicar y medir.

`check` valida la campaña y cuenta los trabajos sin leer datos. `prepare` crea las
vistas de cada ámbito desde la supervisión histórica. `run` ejecuta o reanuda los
trabajos y se detiene si rige el bloqueo de aprendizaje. `sources` publica el
manifiesto de fuentes de un ámbito. `posttraining check|run` valida o ejecuta la etapa
de la matriz de adaptadores sobre una campaña base confirmada. `rl check|run` valida o
ejecuta la etapa de políticas financieras por ventana, con las capacidades del motor que
necesita. `ablation check|run|sources` valida, ejecuta o publica la ablación de modalidades
en inferencia sobre una campaña base confirmada, sin ajustar nada. `throughput` mide en la
GPU, sin pasos de optimizador, el caudal de las familias declaradas y estima las horas de las
variantes indicadas. `extensions` comprueba sin leer datos la declaración preparada de la GRU
candidata, MARS-TITAN y CM-v1 y sus recuentos. `storage` estima el disco de la campaña y de sus
etapas con los recuentos de las vistas y tablas sintéticas, sin leer objetivos ni ajustar.
`budget` proyecta las horas de una campaña con un caudal supuesto o medido y el factor de
caudal necesario para un objetivo de horas. `schedule` muestra las fases de cada ventana de
campaña con sus trabajos, en el orden en que se ejecutan.
"""

import sys

from mars_titan.posttraining import campaign_stage
from mars_titan.simulation import campaign_stage as rl_stage
from mars_titan.training import (
    campaign_budget,
    campaign_extensions,
    campaign_schedule,
    campaign_throughput,
    masked_campaign,
    modality_ablation_stage,
    storage_budget,
)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["throughput"]:
        return campaign_throughput.main(argv[1:])
    if argv[:1] == ["budget"]:
        return campaign_budget.main(argv[1:])
    if argv[:1] == ["schedule"]:
        return campaign_schedule.main(argv[1:])
    if argv[:1] == ["extensions"]:
        return campaign_extensions.main(argv[1:])
    if argv[:1] == ["posttraining"]:
        return campaign_stage.main(argv[1:])
    if argv[:1] == ["rl"]:
        return rl_stage.main(argv[1:])
    if argv[:1] == ["ablation"]:
        return modality_ablation_stage.main(argv[1:])
    if argv[:1] == ["storage"]:
        return storage_budget.main(argv[1:])
    return masked_campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
