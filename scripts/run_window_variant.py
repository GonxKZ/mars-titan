"""Preparar una vista corta o ejecutar una variante experimental de ventanas.

La receta JSON contiene los campos de WindowVariantRecipe. prepare no ajusta modelos.
run conserva el bloqueo de aprendizaje y la reserva exclusiva de CUDA existente.
"""

import argparse
import json
from pathlib import Path

MAX_RECIPE_BYTES = 64 * 1024


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"La receta repite el campo {key}")
        result[key] = value
    return result


def _nonfinite(value):
    raise ValueError(f"La receta no admite el valor no finito {value}")


def _document(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_RECIPE_BYTES + 1)
    if len(data) > MAX_RECIPE_BYTES:
        raise ValueError("La receta supera 64 KiB")
    document = json.loads(data, parse_constant=_nonfinite, object_pairs_hook=_unique)
    if not isinstance(document, dict):
        raise ValueError("La receta debe ser un objeto JSON")
    return document


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        command = commands.add_parser(name)
        command.add_argument("--recipe", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--parent", type=Path, required=True)
            command.add_argument("--catalog", type=Path, required=True)
        else:
            command.add_argument("--manifest", type=Path, required=True)
            command.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        document = _document(args.recipe)
        from mars_titan.training.window_variant_run import (
            WindowVariantRecipe,
            prepare_window_variant_view,
            run_window_variant,
        )

        recipe = WindowVariantRecipe(**document)
    except (OSError, ValueError, TypeError) as error:
        parser.error(f"La receta no es válida: {error}")
    if args.command == "prepare":
        manifest = prepare_window_variant_view(args.parent, args.catalog, args.output, recipe)
        print(json.dumps(dict(status="prepared", manifest=str(manifest), output=str(args.output))))
        return 0

    from mars_titan.training.learning_hold import require_learning_allowed

    require_learning_allowed("el CLI experimental de ventanas independientes")
    from mars_titan.training.checkpoints import StopRequest
    from mars_titan.training.experiment_resources import GpuLease

    with StopRequest() as stop, GpuLease(max_vram=recipe.max_cuda_bytes):
        report = run_window_variant(
            args.manifest, args.output, recipe, resume=args.resume, stop=stop
        )
    summary = dict(
        status=report["status"], global_step=report.get("global_step", 0), output=str(args.output)
    )
    print(json.dumps(summary, ensure_ascii=False))
    if report["status"] not in {"completed", "paused", "partial"}:
        raise ValueError("El runner devolvió un estado de ejecución desconocido")
    return 0 if report["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())
