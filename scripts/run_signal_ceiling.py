"""Techo de señal con anomalías conocidas en los cuatro orígenes de la criba (#747, fase 0b).

Sin `--run` comprueba la declaración e imprime sus orígenes, modelos y objetivos sin leer datos.
Con `--run` comprueba la protección del aprendizaje antes de abrir nada, verifica las cohortes
de la criba leyendo solo sus manifiestos, ajusta y mide cada origen y escribe `report.json`.
Un origen terminado con la misma petición se reutiliza, así que repetir la orden continúa un
corte. `--cohorts` es la carpeta con un `AÑO/manifest.json` por origen.
"""

import argparse
import json
from pathlib import Path

from mars_titan.training.learning_hold import require_learning_allowed

ROOT = Path(__file__).resolve().parents[1]
DECLARATION = ROOT / "configs" / "experiments" / "signal-ceiling-v1.json"


def read_declaration(path):
    """Declaración del estudio comprobada con sus orígenes, características y modelos."""
    from mars_titan.training import signal_ceiling

    return signal_ceiling.checked_declaration(json.loads(path.read_text(encoding="utf-8")))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--declaration", type=Path, default=DECLARATION)
    parser.add_argument("--cohorts", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.run and (args.cohorts is None or args.output is None):
        raise SystemExit("La medida necesita --cohorts y --output")
    if args.run:
        require_learning_allowed("el techo de señal con anomalías conocidas")
    declaration = read_declaration(args.declaration)
    if not args.run:
        plan = dict(
            origins=declaration["screening"]["origins"],
            models=sorted(declaration["models"]),
            targets=sorted(declaration["targets"]),
        )
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return 0

    from mars_titan.data.storage import atomic_json
    from mars_titan.evaluation import signal_ceiling_report
    from mars_titan.training import screening_cohorts, signal_ceiling

    screening_cohorts.verify(args.cohorts)
    origins = [
        signal_ceiling.run_origin(
            args.cohorts / str(origin["year"]) / "manifest.json",
            origin,
            declaration,
            args.output / "origins",
        )
        for origin in declaration["screening"]["origins"]
    ]
    report = signal_ceiling_report.report(origins, declaration["screening"]["inference"])
    atomic_json(args.output / "report.json", report)
    print(json.dumps(dict(status="completed", output=str(args.output)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
