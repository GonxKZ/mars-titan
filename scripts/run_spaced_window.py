"""Ejecutar una variante del diagnóstico E2 con parámetros seleccionados y congelados."""

import argparse
import json
from pathlib import Path

from mars_titan.training.learning_hold import require_learning_allowed


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("view", "anchor", "source-manifest", "study", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda:0"), default="cuda:0")
    parser.add_argument("--indices", type=Path)
    args = parser.parse_args(argv)
    require_learning_allowed("la orden de evaluación congelada E2")

    from mars_titan.training.checkpoints import StopRequest
    from mars_titan.training.spaced_window import run_spaced_window

    with StopRequest() as stop:
        report = run_spaced_window(
            args.view,
            args.anchor,
            args.source_manifest,
            args.study,
            arm=args.arm,
            output=args.output,
            device=args.device,
            indices=args.indices,
            stop=stop,
        )
    if report["status"] not in ("completed", "paused"):
        raise ValueError("La ventana E2 no devolvió un estado completado o pausado")
    print(
        json.dumps(
            dict(status=report["status"], run_id=report["run_id"], output=str(args.output)),
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
