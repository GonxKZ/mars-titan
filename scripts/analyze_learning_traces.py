"""Analizar carpetas de trazas de aprendizaje y escribir el informe JSON y Markdown.

`--traces` admite varias carpetas del registrador y `--root` busca todas las carpetas
`traces/` bajo una salida de campaña o de prueba. El informe sigue las reglas fijas de
`mars_titan.learning_traces.analysis` y no modifica las trazas.
"""

import argparse
import json
from pathlib import Path

from mars_titan.data.storage import atomic_json
from mars_titan.learning_traces.analysis import analyze, discover, markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--traces", type=Path, nargs="+", help="Carpetas del registrador")
    sources.add_argument("--root", type=Path, help="Carpeta en la que buscar trazas")
    parser.add_argument("--output", type=Path, required=True, help="Informe JSON")
    parser.add_argument("--markdown", type=Path, help="Informe Markdown")
    parser.add_argument("--title", default="Análisis de trazas de aprendizaje")
    parser.add_argument("--note", help="Párrafo inicial del informe Markdown")
    parser.add_argument("--limit", type=int, default=4096)
    args = parser.parse_args(argv)
    folders = discover(args.root, limit=args.limit) if args.root else args.traces
    if not folders:
        parser.error("No hay carpetas de trazas")
    report = analyze(folders, root=args.root)
    if args.root is not None:
        # Rutas relativas: el informe no depende de dónde estaba la salida.
        for run in report["runs"].values():
            run["folder"] = str(Path(run["folder"]).relative_to(args.root))
    atomic_json(args.output, report)
    if args.markdown:
        args.markdown.write_text(markdown(report, title=args.title, note=args.note))
    print(json.dumps(dict(runs=len(report["runs"]), output=str(args.output))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
