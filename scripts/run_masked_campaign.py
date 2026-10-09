"""Orden única de la campaña con máscaras: comprobar, preparar, ejecutar, publicar y medir.

`check` valida la campaña y cuenta los trabajos sin leer datos. `prepare` crea las
vistas de cada ámbito desde la supervisión histórica. `run` ejecuta o reanuda los
trabajos y se detiene si rige el bloqueo de aprendizaje. `sources` publica el
manifiesto de fuentes de un ámbito. `throughput` mide el caudal neuronal en la GPU sin
pasos de optimizador y estima las horas de las variantes indicadas.
"""

import sys

from mars_titan.training import campaign_throughput, masked_campaign


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["throughput"]:
        return campaign_throughput.main(argv[1:])
    return masked_campaign.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
