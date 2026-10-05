"""Preparación lateral y comparación técnica de documentos con entradas idénticas."""

import argparse
import json
import platform
import resource
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

from .document_index import DocumentIndex, prepare_document_index
from .storage import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Conservar documentos sin recodificar")
    prepare.add_argument("--source", action="append", required=True, metavar="ACTIVO=MANIFIESTO")
    prepare.add_argument("--cache", type=Path, required=True)
    prepare.add_argument("--encoder", required=True)
    prepare.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare", help="Medir media y selección sobre la misma ventana")
    compare.add_argument("--index", type=Path, required=True)
    compare.add_argument("--cache", type=Path, required=True)
    compare.add_argument("--asset", required=True)
    compare.add_argument("--start", type=datetime.fromisoformat, required=True)
    compare.add_argument("--end", type=datetime.fromisoformat, required=True)
    compare.add_argument("--selected", type=int, default=8)
    compare.add_argument("--repeats", type=int, default=5)
    compare.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.command == "prepare":
        sources = {}
        for source in args.source:
            asset, separator, path = source.partition("=")
            if not separator or not path or asset in sources:
                parser.error("Cada fuente requiere un activo distinto y su manifiesto")
            sources[asset] = Path(path)
        result = prepare_document_index(
            sources, args.output, cache_path=args.cache, encoder_sha256=args.encoder
        )
    else:
        if not 1 <= args.repeats <= 100:
            parser.error("Las repeticiones deben estar entre una y cien")
        durations = []
        with DocumentIndex(args.index, cache_path=args.cache) as reader:
            # Una pasada independiente calienta páginas y las dos entradas de la caché Arrow.
            reference = reader.compare(args.asset, args.start, args.end, selected=args.selected)
            for _ in range(args.repeats):
                begin = time.perf_counter()
                observed = reader.compare(args.asset, args.start, args.end, selected=args.selected)
                durations.append(time.perf_counter() - begin)
                if observed != reference:
                    raise ValueError("La misma ventana ha cambiado entre repeticiones")
            result = dict(
                index_sha256=reader.identity,
                asset_id=args.asset,
                start=args.start.isoformat(),
                end=args.end.isoformat(),
                comparison=reference,
                measurement=dict(
                    repetitions=args.repeats,
                    warmup=1,
                    query_seconds=durations,
                    p50_seconds=float(np.quantile(durations, 0.5)),
                    p95_seconds=float(np.quantile(durations, 0.95)),
                    mentions_per_second=reference["examined_mentions"] / float(np.mean(durations)),
                    wall_seconds=time.perf_counter() - started,
                    peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                    parquet_bytes=reader.data.stat().st_size,
                    gpu_used=False,
                    energy_joules=None,
                    total_cost=None,
                ),
                environment=dict(
                    python=platform.python_version(),
                    numpy=np.__version__,
                    pyarrow=pa.__version__,
                    machine=platform.machine(),
                    platform=platform.platform(),
                ),
            )
        if args.output:
            if args.output.exists():
                raise ValueError("La salida de la medición ya existe")
            atomic_json(args.output, result)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
