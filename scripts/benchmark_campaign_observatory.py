"""Medir el recorrido de lectura incremental y exportación con sus fuentes reales."""

import argparse
import json
import platform
import resource
import statistics
import tempfile
import time
from pathlib import Path

from mars_titan.data.storage import atomic_json
from mars_titan.observatory.collector import Collector, write_pages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/observatory/campaigns.json"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sources = json.loads(args.config.read_text())["sources"]
    measured = []
    started = time.perf_counter()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        with Collector(args.root, root / "cache.sqlite") as collector:
            for index in range(6):
                before = time.perf_counter()
                snapshot = collector.collect(sources)
                write_pages(snapshot, root / "public")
                measured.append(
                    dict(
                        cold=index == 0,
                        seconds=time.perf_counter() - before,
                        bytes_read=collector.bytes_read,
                        runs=len(snapshot["runs"]),
                    )
                )
    warm = [row["seconds"] for row in measured[1:]]
    atomic_json(
        args.output,
        dict(
            python=platform.python_version(),
            platform=platform.platform(),
            workers=1,
            repetitions=5,
            warmup=1,
            measurements=measured,
            warm_median_seconds=statistics.median(warm),
            warm_stdev_seconds=statistics.stdev(warm),
            process_seconds=time.perf_counter() - started,
            peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            gpu_used=False,
            concurrent_scientific_workload=True,
            limits="Lectura lógica de JSON. No mide contadores físicos de disco ni energía.",
        ),
    )


if __name__ == "__main__":
    main()
