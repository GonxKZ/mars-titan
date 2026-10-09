"""Orden única de la campaña con máscaras: comprobar, preparar, ejecutar, publicar y medir.

`check` valida la campaña y cuenta los trabajos sin leer datos. `prepare` crea las
vistas de cada ámbito desde la supervisión histórica. `run` ejecuta o reanuda los
trabajos y se detiene si rige el bloqueo de aprendizaje. `sources` publica el
manifiesto de fuentes de un ámbito. `posttraining check|run` valida o ejecuta la etapa
de la matriz de adaptadores sobre una campaña base confirmada. `rl check|run` valida o
ejecuta la etapa de políticas financieras por ventana, con las capacidades del motor que
necesita. `rl-report` agrega sus salidas confirmadas en el informe financiero.
`ablation check|run|sources` valida, ejecuta o publica la ablación de modalidades en inferencia
sobre una campaña base confirmada, sin ajustar nada. `throughput` mide en la GPU, sin pasos de
optimizador, el caudal de las familias declaradas y estima las horas de las variantes
indicadas. `extensions` comprueba sin leer datos la declaración preparada de la GRU candidata,
MARS-TITAN y CM-v1 y sus recuentos. `storage` estima el disco de la campaña y de sus etapas
con los recuentos de las vistas y tablas sintéticas, sin leer objetivos ni ajustar.
"""

import importlib
import sys

# Cada orden importa solo su módulo. Así `rl` no carga el código de otras etapas, como los
# mundos sintéticos de experimentos anteriores que alcanzan los adaptadores.
COMMANDS = {
    "throughput": "mars_titan.training.campaign_throughput",
    "extensions": "mars_titan.training.campaign_extensions",
    "posttraining": "mars_titan.posttraining.campaign_stage",
    "rl": "mars_titan.simulation.campaign_stage",
    "rl-report": "mars_titan.simulation.stage_report",
    "ablation": "mars_titan.training.modality_ablation_stage",
    "storage": "mars_titan.training.storage_budget",
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] and argv[0] in COMMANDS:
        return importlib.import_module(COMMANDS[argv[0]]).main(argv[1:])
    return importlib.import_module("mars_titan.training.masked_campaign").main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
