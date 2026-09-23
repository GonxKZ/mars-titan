"""Medición local sobre todas las decisiones admisibles de AA, sin entrenamiento."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from mars_titan.data.charts import chart_png
from mars_titan.data.cohort_contexts import MacroVectors
from mars_titan.data.cohort_samples import materialize_cohort_asset
from mars_titan.data.embeddings import EmbeddingCache, FrozenEncoders
from mars_titan.data.samples import materialize_samples
from mars_titan.data.storage import atomic_json, sha256
from mars_titan.data.temporal import MarketClock

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--variant", choices=["reference", "streamed"], required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--context-note", default="Concurrencia no auditada")
args = parser.parse_args()
if args.output.exists():
    raise ValueError("La salida debe ser nueva")
args.output.mkdir(parents=True)
started = time.perf_counter()
torch.set_num_threads(4)
source = Path("data/processed/original-audited-20260923/prepared/US/AA").resolve()
macro = Path("data/processed/phase1/macro-US.parquet")
clock = MarketClock("US", "1990-01-01", "2026-01-01")
encoders = FrozenEncoders()
news = pq.ParquetFile(source / "news/news.parquet").read_row_group(0).to_pylist()
prices = pq.read_table(source / "prices.parquet").to_pandas()
encoders.text(news[0]["text"])
encoders.images([chart_png(prices[["open", "high", "low", "close"]].to_numpy(), end_index=63)])
torch.cuda.synchronize()
warmup = time.perf_counter() - started
cache = EmbeddingCache(args.output / "cache.sqlite")
legacy = args.output / "legacy/US/AA"
if args.variant == "reference":
    legacy.mkdir(parents=True)
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["artifacts"] = {}
    for name, original in (
        ("prices.parquet", "prices.parquet"),
        ("fundamentals.parquet", "fundamentals.parquet"),
        ("news.parquet", "news/news.parquet"),
    ):
        (legacy / name).symlink_to(source / original)
        manifest["artifacts"][name] = sha256(source / original)
    atomic_json(legacy / "manifest.json", manifest)
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
try:
    if args.variant == "reference":
        report = materialize_samples(
            args.output / "legacy",
            macro,
            args.output / "encoded",
            {"assets": [{"symbol": "AA"}]},
            clock,
            encoders,
            cache,
            company_factors=True,
        )["assets"][0]
    else:
        contexts = MacroVectors(macro)
        report = materialize_cohort_asset(
            source,
            args.output / "encoded/US/AA",
            clock,
            contexts,
            encoders,
            cache,
            cohort="original_audited",
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
finally:
    cache.close()
atomic_json(
    args.output / "measurement.json",
    dict(
        variant=args.variant,
        elapsed_seconds=elapsed,
        setup_and_warmup_seconds=warmup,
        whole_process_internal_seconds=time.perf_counter() - started,
        samples=report["samples"],
        samples_per_second=report["samples"] / elapsed,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
        peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
        source_manifest_sha256=sha256(source / "manifest.json"),
        macro_sha256=sha256(macro),
        samples_sha256=report["samples_sha256"],
        encoders=encoders.spec,
        concurrent_workloads=args.context_note,
        measurement_code_sha256=sha256(Path(__file__)),
        omp_threads=os.environ.get("OMP_NUM_THREADS"),
        torch_threads=torch.get_num_threads(),
    ),
)
print(json.dumps({"variant": args.variant, "samples": report["samples"], "seconds": elapsed}))
