"""Coste por fase de los rangos NVTX apagados y encendidos sin herramienta conectada.

Mide con `timeit` un bloque `with` vacío en cuatro casos: sin rango, con `phase` apagado,
con `phase` encendido y con `range_push` y `range_pop` directos de `torch.cuda.nvtx`. Sin
Nsight Systems conectado, las llamadas de NVTX no registran nada, así que el caso encendido
mide solo el cruce a la biblioteca. Cada caso se repite y se publica la mediana y el mínimo
por llamada.

Uso: nvtx_cost.py SALIDA.json [--number 200000] [--repeat 15]
"""

import argparse
import json
import os
import platform
import statistics
import sys
import timeit
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    parser.add_argument("--number", type=int, default=200_000)
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    if os.environ.get("MARS_TITAN_NVTX", "0") != "0":
        raise SystemExit("La medida importa el módulo con los rangos apagados")
    import torch

    from mars_titan import nvtx_ranges

    nvtx = torch.cuda.nvtx

    def bare():
        pass

    def disabled():
        with nvtx_ranges.phase("titans.observe"):
            pass

    def enabled():
        with nvtx_ranges._Range("titans.observe"):
            pass

    def direct():
        nvtx.range_push("titans.observe")
        nvtx.range_pop()

    cases = dict(bare=bare, disabled=disabled, enabled=enabled, direct=direct)
    results = {}
    for name, function in cases.items():
        times = timeit.repeat(function, number=args.number, repeat=args.repeat)
        per_call = [value / args.number * 1e9 for value in times]
        results[name] = dict(median_ns=statistics.median(per_call), min_ns=min(per_call))
    report = dict(
        python=platform.python_version(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        number=args.number,
        repeat=args.repeat,
        load_average=os.getloadavg(),
        per_call=results,
        disabled_over_bare_ns=results["disabled"]["median_ns"] - results["bare"]["median_ns"],
        enabled_over_bare_ns=results["enabled"]["median_ns"] - results["bare"]["median_ns"],
    )
    args.output.write_text(json.dumps(report, indent=1) + "\n")
    json.dump(report, sys.stdout)


if __name__ == "__main__":
    main()
