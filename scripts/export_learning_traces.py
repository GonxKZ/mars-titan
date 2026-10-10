"""Convertir las trazas del registrador de aprendizaje en un paquete del observatorio.

Lee el manifiesto y las partes Parquet con `read_traces`, que comprueba esquema y
numeración, y escribe un paquete binario con `write_trace_bundle`. No modifica la carpeta
de origen. La cadencia y el paso en el que el registrador agotó su presupuesto pasan al
paquete, de modo que la página no confunde ese corte con el final del entrenamiento. El
paquete resultante se sirve con `serve_observatory.py --traces-dir`.
"""

import argparse
from pathlib import Path

from mars_titan.learning_traces.recorder import read_traces
from mars_titan.observatory.traces import series_from_learning_traces, write_trace_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True, help="Carpeta del registrador")
    parser.add_argument("--output", type=Path, required=True, help="Carpeta de paquetes")
    parser.add_argument("--name", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--provenance", choices=("measured", "fixture"), default="measured")
    arguments = parser.parse_args()
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
        x_unit="optimizer_step",
        series=series,
        cadence=manifest.get("config", {}).get("every"),
        truncated_at=manifest.get("exhausted_at_step"),
    )
    print(f"{len(bundle['series'])} series y {bundle['blob_bytes']} bytes en {arguments.output}")


if __name__ == "__main__":
    main()
