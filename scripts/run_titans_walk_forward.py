"""Ajustar una ventana walk-forward de Titans-MAC y escribir sus predicciones por fila.

Comprueba el bloqueo de aprendizaje antes de reservar la GPU o abrir la vista. Con
`cuda:0` toma la reserva exclusiva de la carga científica. Termina con código 0 si la
ventana queda completada y con 3 si se pausa en una barrera confirmada.
"""

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from mars_titan.models.titans.financial import VARIANTS
from mars_titan.training.checkpoints import StopRequest
from mars_titan.training.experiment_resources import GpuLease
from mars_titan.training.learning_hold import require_learning_allowed
from mars_titan.training.titans_walk_forward import run_titans_window


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--view", type=Path, required=True, help="manifest.json de la ventana")
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--window", required=True, help="Identificador, por ejemplo fold-000")
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--search-case", help="Caso de búsqueda, si la receta los declara")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--indices", type=Path, help="Índices compartidos por la ventana")
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    args = parser.parse_args(argv)
    require_learning_allowed("run_titans_walk_forward.py")
    # El entrenador exige la ruta no fusionada de atención y la registra en su identidad.
    torch.backends.mha.set_fastpath_enabled(False)
    lease = GpuLease() if args.device == "cuda:0" else nullcontext()
    with StopRequest() as stop, lease:
        report = run_titans_window(
            args.view,
            args.protocol,
            args.window,
            args.recipe,
            variant=args.variant,
            seed=args.seed,
            output=args.output,
            device=args.device,
            indices=args.indices,
            stop=stop,
            search_case=args.search_case,
        )
    summary = dict(status=report["status"], run_id=report.get("run_id"), output=str(args.output))
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
