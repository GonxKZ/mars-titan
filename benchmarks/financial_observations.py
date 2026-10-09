"""Medir la lectura cronológica por observación frente a la lectura por bloques de activos.

El modo técnico genera un corpus con anchuras reales y compara ambos lectores sobre la
misma fase. El modo `--edition` solo lee archivos codificados ya confirmados y mide la
decodificación de un grupo Parquet, sin escribir en esa ubicación.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow
import pyarrow.parquet as pq

from mars_titan.data.input_policy import HISTORICAL_MASKED
from mars_titan.memory import financial_observations as api
from mars_titan.training.cohort_contract import representation_identity
from mars_titan.training.corpus_inputs import CorpusDataset, _price_contexts
from tests.training.chronological_fixture import chronological_corpus, phases

REAL_WIDTHS = (384, 512, 26, 140)


def _digest(events):
    digest, rows = hashlib.sha256(), 0
    for event in events:
        digest.update(repr((event.at, event.labels, event.close_phase)).encode())
        for batch in event.inputs:
            for index, identity in enumerate(batch["sample_ids"]):
                digest.update(identity.encode())
                digest.update(batch["presence"][index].tobytes())
                for name in sorted(batch["inputs"]):
                    digest.update(batch["inputs"][name][index].tobytes())
                rows += 1
    return digest.hexdigest(), rows


def measure(stream, *, block_rows, repeats):
    """Recorrer la fase completa con cada lector y comprobar la misma huella."""
    result = {}
    for name, read in (
        ("per_observation", lambda: stream.events()),
        ("asset_blocks", lambda: stream.batched_events(block_rows=block_rows)),
    ):
        seconds, digests = [], set()
        for _ in range(repeats):
            start = time.perf_counter()
            digest, rows = _digest(read())
            seconds.append(time.perf_counter() - start)
            digests.add(digest)
        if len(digests) != 1:
            raise ValueError("Un lector no es determinista")
        result[name] = dict(
            seconds=seconds,
            median_seconds=statistics.median(seconds),
            rows=rows,
            microseconds_per_observation=1e6 * statistics.median(seconds) / rows,
            stream_sha256=digests.pop(),
        )
    if result["per_observation"]["stream_sha256"] != result["asset_blocks"]["stream_sha256"]:
        raise ValueError("Los lectores no entregan el mismo recorrido")
    result["speedup"] = (
        result["per_observation"]["median_seconds"] / result["asset_blocks"]["median_seconds"]
    )
    return result


def technical(output, *, assets, first, last, group_size, block_rows, repeats):
    manifest = chronological_corpus(
        Path(output) / "corpus",
        assets=assets,
        first=first,
        last=last,
        group_size=group_size,
        widths=REAL_WIDTHS,
    )
    dataset = CorpusDataset(manifest, input_policy=HISTORICAL_MASKED)
    report = {}
    for phase in phases(warmup=first, train_decisions=first, validation_warmup="2022-12-01"):
        index = api.prepare_observation_index(
            dataset, Path(output) / f"index-{phase.partition}", phase=phase
        )
        stream = api.FinancialObservationSource(dataset, index)
        report[phase.partition] = measure(stream, block_rows=block_rows, repeats=repeats)
    return dict(
        workload=dict(
            assets=assets,
            first=first,
            last=last,
            group_rows=group_size,
            block_rows=block_rows,
            widths=dict(
                zip(("news", "charts", "concepts", "indicators"), REAL_WIDTHS, strict=True)
            ),
            repeats=repeats,
        ),
        phases=report,
    )


def edition(root, symbols, *, groups):
    """Decodificar grupos reales con las reglas del lector, solo con acceso de lectura."""
    root = Path(root)
    prepared = Path(json.loads((root / "configuration.json").read_text())["prepared_root"])
    result = []
    for symbol in symbols:
        folder = root / "samples/US" / symbol
        receipt = json.loads((folder / "manifest.json").read_text())
        path = folder / "samples.parquet"
        reader = SimpleNamespace(
            _file=lambda asset, kind, path=path: path,
            temporals={},
            temporal=None,
            cohort=None,
            masked=True,
            manifest={
                "representation": representation_identity(receipt, input_policy=HISTORICAL_MASKED)
            },
            cache_sample_tables=False,
            verified={path: None},
        )
        with pq.ParquetFile(path) as file:
            sizes = [file.metadata.row_group(g).num_rows for g in range(file.num_row_groups)]
            seconds = []
            for group in range(min(groups, file.num_row_groups)):
                start = time.perf_counter()
                _, _, ends, *_ = CorpusDataset._sample_group(reader, {"market": "US"}, file, group)
                seconds.append(time.perf_counter() - start)
        table = pq.read_table(prepared / "US" / symbol / "prices.parquet")
        prices = np.column_stack(
            [table[c].to_numpy() for c in ("open", "high", "low", "close", "volume")]
        )
        start = time.perf_counter()
        for row in range(len(ends)):
            _price_contexts(prices, ends[[row]], 64)
        context = (time.perf_counter() - start) / len(ends)
        decode = statistics.median(seconds)
        result.append(
            dict(
                symbol=symbol,
                rows=sum(sizes),
                row_groups=len(sizes),
                rows_per_group=max(sizes),
                group_decode_seconds=seconds,
                per_observation_seconds=decode + context,
                asset_blocks_seconds=decode / max(sizes) + context,
            )
        )
    return result


def environment():
    return dict(
        recorded_at_utc=datetime.now(UTC).isoformat(),
        python=platform.python_version(),
        platform=platform.platform(),
        processor=platform.processor() or platform.machine(),
        cpu_count=os.cpu_count(),
        omp_threads=os.environ.get("OMP_NUM_THREADS"),
        pyarrow=pyarrow.__version__,
        numpy=np.__version__,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assets", type=int, default=16)
    parser.add_argument("--first", default="2022-07-01")
    parser.add_argument("--last", default="2023-03-31")
    parser.add_argument("--group-rows", type=int, default=128)
    parser.add_argument("--block-rows", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--edition", type=Path)
    parser.add_argument("--symbols", default="AA,AFMC,AROW")
    parser.add_argument("--groups", type=int, default=8)
    arguments = parser.parse_args()
    if arguments.output.exists():
        raise FileExistsError("La salida del benchmark debe ser nueva")
    report = dict(
        technical=technical(
            arguments.output,
            assets=arguments.assets,
            first=arguments.first,
            last=arguments.last,
            group_size=arguments.group_rows,
            block_rows=arguments.block_rows,
            repeats=arguments.repeats,
        )
    )
    if arguments.edition is not None:
        report["edition"] = edition(
            arguments.edition, arguments.symbols.split(","), groups=arguments.groups
        )
    report["environment"] = environment()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
