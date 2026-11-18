"""Medir la apertura del corpus hasta el primer lote cronológico verificado."""

import argparse
import json
import os
import resource
import threading
import time
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory.financial_observations import FinancialObservationSource
from mars_titan.models.titans.financial_inputs import validated_cpu_batches
from mars_titan.training import corpus_inputs


def measure(view, index, block_rows):
    hashed = 0
    bytes_hashed = 0
    lock = threading.Lock()
    digest = corpus_inputs.sha256

    def counted_digest(path):
        nonlocal hashed, bytes_hashed
        result = digest(path)
        if path.suffix == ".parquet":
            with lock:
                hashed += 1
                bytes_hashed += path.stat().st_size
        return result

    start = time.perf_counter()
    with patch.object(corpus_inputs, "sha256", counted_digest):
        dataset = corpus_inputs.CorpusDataset(view, input_policy=HISTORICAL_MASKED)
        open_seconds = time.perf_counter() - start
        source = FinancialObservationSource(dataset, index)
        if source.phase.partition != "train":
            raise ValueError("La medida necesita un índice de ajuste")
        spec = source.specification()
        with closing(source.batched_events(block_rows=block_rows)) as events:
            first = next(event for event in events if event.inputs)
            inputs = validated_cpu_batches(first.inputs, spec)

    return dict(
        cohort_sha256=dataset.identity,
        index_sha256=source.identity,
        shared=bool(os.environ.get(corpus_inputs.DIGEST_CACHE_ENV)),
        parquet_files_hashed=hashed,
        parquet_bytes_hashed=bytes_hashed,
        corpus_open_seconds=open_seconds,
        first_batch_seconds=time.perf_counter() - start,
        first_at=first.at,
        input_sha256=[batch.input_digest for batch in inputs],
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("view", type=Path, help="Manifiesto de la cohorte")
    parser.add_argument("index", type=Path, help="Manifiesto del índice de ajuste")
    parser.add_argument("--block-rows", type=int, default=512, help="Filas por bloque de entrada")
    args = parser.parse_args()
    print(json.dumps(measure(args.view, args.index, args.block_rows), ensure_ascii=False))
