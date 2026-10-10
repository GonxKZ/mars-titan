"""Convertir las trazas del registrador de aprendizaje en paquetes del observatorio.

Con `--traces` se convierte una carpeta del registrador. Lee el manifiesto y las partes
Parquet con `read_traces`, que comprueba esquema y numeración, y escribe un paquete binario
con `write_trace_bundle`. Con `--root` se convierten todas las carpetas `traces/` de una
salida de campaña, y con `--watch` se repite cada tantos segundos y solo se vuelven a leer
las carpetas con partes nuevas. Ninguno de los dos modos modifica las trazas de origen. La
cadencia y el paso en el que el registrador agotó su presupuesto pasan al paquete, de modo
que la página no confunde ese corte con el final del entrenamiento. Los paquetes se sirven
con `serve_observatory.py --traces-dir`, que avisa a la página cuando cambia `index.json`.
"""

import argparse
import json
import signal
import time
from pathlib import Path

from mars_titan.learning_traces.recorder import read_traces
from mars_titan.observatory.traces import (
    series_from_learning_traces,
    sync_learning_traces,
    write_trace_bundle,
)


def export_folder(arguments):
    manifest, table = read_traces(arguments.traces)
    columns = table.to_pydict()
    series = series_from_learning_traces(
        columns["step"],
        columns["phase"],
        columns["metric"],
        columns["group"],
        columns["stat"],
        columns["value"],
    )
    bundle = write_trace_bundle(
        arguments.output,
        arguments.name,
        run_id=arguments.run_id,
        attempt_id=arguments.attempt_id,
        model_id=arguments.model_id,
        provenance=arguments.provenance,
        x_unit=manifest["step_unit"],
        series=series,
        cadence=manifest["config"]["every"],
        truncated_at=manifest["exhausted_at_step"],
    )
    print(f"{len(bundle['series'])} series y {bundle['blob_bytes']} bytes en {arguments.output}")


def sync_root(arguments):
    stopped = False

    def stop(*_):
        nonlocal stopped
        stopped = True

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    seen = {}
    while not stopped:
        started = time.monotonic()
        result = sync_learning_traces(
            arguments.root,
            arguments.output,
            limit=arguments.limit,
            seen=seen,
            provenance=arguments.provenance,
        )
        print(
            json.dumps(
                dict(
                    found=result["found"],
                    written=len(result["written"]),
                    skipped=result["skipped"],
                    seconds=round(time.monotonic() - started, 3),
                ),
                ensure_ascii=False,
            ),
            flush=True,
        )
        if arguments.watch is None:
            break
        while not stopped and time.monotonic() - started < arguments.watch:
            time.sleep(0.5)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--traces", type=Path, help="Carpeta del registrador")
    sources.add_argument("--root", type=Path, help="Salida en la que buscar carpetas traces/")
    parser.add_argument("--output", type=Path, required=True, help="Carpeta de paquetes")
    parser.add_argument("--name")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt-id")
    parser.add_argument("--model-id")
    parser.add_argument("--provenance", choices=("measured", "fixture"), default="measured")
    parser.add_argument("--limit", type=int, default=256, help="Carpetas recientes con --root")
    parser.add_argument("--watch", type=float, help="Segundos entre recorridos con --root")
    arguments = parser.parse_args()
    if arguments.traces is not None:
        missing = [
            name
            for name in ("name", "run_id", "attempt_id", "model_id")
            if getattr(arguments, name) is None
        ]
        if missing:
            parser.error("--traces necesita --name, --run-id, --attempt-id y --model-id")
        export_folder(arguments)
    else:
        if arguments.watch is not None and not 1 <= arguments.watch <= 3600:
            parser.error("--watch admite entre 1 y 3600 segundos")
        sync_root(arguments)


if __name__ == "__main__":
    main()
