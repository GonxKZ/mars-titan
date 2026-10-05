"""Control sintético de preparación documental. No representa evidencia histórica."""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from mars_titan.data.cohort_news import write_cohort_news
from mars_titan.data.document_cli import main as document_cli
from mars_titan.data.document_index import prepare_document_index
from mars_titan.data.embeddings import EmbeddingCache
from mars_titan.data.storage import atomic_json
from mars_titan.data.temporal import MarketClock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=int, default=8)
    parser.add_argument("--documents", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.assets <= 128 or not 2 <= args.documents <= 4096:
        parser.error("El control admite 1 a 128 activos y 2 a 4096 documentos por activo")
    if args.output.exists():
        parser.error("El control necesita un directorio nuevo")
    raw = args.output / "synthetic-source"
    raw.mkdir(parents=True)
    clock = MarketClock("US", "2023-01-01", "2024-01-01")
    start = datetime(2023, 7, 5, 12, tzinfo=UTC)
    encoder = "0" * 64
    cache_path = args.output / "synthetic-vectors.sqlite"
    cache, sources = EmbeddingCache(cache_path), {}
    try:
        for asset in range(args.assets):
            symbol = f"CONTROL{asset:03}"
            path = raw / f"{symbol}.jsonl"
            with path.open("w") as stream:
                for document in range(args.documents):
                    row = dict(
                        Date=(start + timedelta(minutes=document)).isoformat(),
                        Stock_symbol=symbol,
                        Article_title=f"Documento de control {document}",
                        Article="Contenido sintético para probar el índice, sin hecho económico.",
                        Url=f"https://example.org/synthetic-control/{document}",
                    )
                    stream.write(json.dumps(row) + "\n")
            output = args.output / "prepared" / symbol
            write_cohort_news(
                raw,
                [path.name],
                output,
                symbol=symbol,
                clock=clock,
                cohort="original_audited",
                batch_rows=128,
            )
            sources[f"US/{symbol}"] = output / "manifest.json"
            for document, row in enumerate(pq.read_table(output / "news.parquet").to_pylist()):
                identity = dict(
                    encoder=encoder,
                    kind="news",
                    content=row["content_hash"],
                    policy=row["availability_rule"],
                )
                if cache.get(identity) is None:
                    cache.put(identity, np.full(384, document / args.documents, dtype=np.float32))
    finally:
        cache.close()
    prepare_document_index(
        sources,
        args.output / "index",
        cache_path=cache_path,
        encoder_sha256=encoder,
        batch_rows=128,
    )
    atomic_json(
        args.output / "control.json",
        dict(
            kind="synthetic_document_control",
            historical_evidence=False,
            learned_encoder=False,
            assets=args.assets,
            documents_per_asset=args.documents,
            vector_width=384,
        ),
    )
    document_cli(
        [
            "compare",
            "--index",
            str(args.output / "index/manifest.json"),
            "--cache",
            str(cache_path),
            "--asset",
            "US/CONTROL000",
            "--start",
            start.isoformat(),
            "--end",
            (start + timedelta(minutes=args.documents)).isoformat(),
            "--selected",
            "8",
            "--repeats",
            str(args.repeats),
            "--output",
            str(args.output / "comparison.json"),
        ]
    )


if __name__ == "__main__":
    main()
