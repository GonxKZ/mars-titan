"""Reparto nativo de la memoria viva de DuckDB en el pico de una captura de memray.

Toma las asignaciones vivas en el máximo de la captura cuya pila de Python pasa por una
función dada (por defecto `_ordered_parquet`, la ordenación de `corpus_source.py`) y las
clasifica por la primera familia de marcos nativos que aparece en su pila: lectura de los
metadatos Parquet, resto del lector Parquet o gestor de bloques de DuckDB. Solo el gestor
de bloques está acotado por `memory_limit`. Necesita una captura hecha con `--native`.

Uso: duckdb_native_split.py CAPTURA.bin SALIDA.json [--function _ordered_parquet]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from memray import FileReader

FAMILIES = (
    ("metadatos Parquet (FileMetaData::read)", ("FileMetaData::read",)),
    ("lector Parquet (resto)", ("ParquetReader",)),
    ("gestor de bloques de DuckDB", ("BufferManager", "BlockHandle", "BufferPool")),
)


def split(capture, function):
    sizes, counts = Counter(), Counter()
    for record in FileReader(capture).get_high_watermark_allocation_records(merge_threads=True):
        if not any(name == function for name, _, _ in record.stack_trace()):
            continue
        native = " ".join(name for name, _, _ in record.native_stack_trace())
        family = next(
            (label for label, keys in FAMILIES if any(key in native for key in keys)), "otros"
        )
        sizes[family] += record.size
        counts[family] += record.n_allocations
    total = sum(sizes.values())
    if not total:
        raise SystemExit(f"No hay asignaciones vivas bajo {function} en el pico")
    return dict(
        function=function,
        bytes=total,
        split=[
            dict(family=label, bytes=size, share=size / total, allocations=counts[label])
            for label, size in sizes.most_common()
        ],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("capture", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--function", default="_ordered_parquet")
    args = parser.parse_args()
    report = dict(capture=args.capture.name, **split(args.capture, args.function))
    args.output.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
