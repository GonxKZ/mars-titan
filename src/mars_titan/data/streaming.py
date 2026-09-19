"""Lectura acotada y ventanas de precios calculadas al consumir cada muestra."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .storage import sha256


def assigned(values: list, *, worker: int = 0, workers: int = 1) -> list:
    if workers < 1 or not 0 <= worker < workers:
        raise ValueError("Invalid worker partition")
    return values[worker::workers]


def price_features(window: np.ndarray) -> np.ndarray:
    if window.ndim != 2 or window.shape[1] != 5 or not np.isfinite(window).all():
        raise ValueError("Invalid price window")
    if (window[:, :4] <= 0).any() or (window[:, 4] < 0).any():
        raise ValueError("Invalid OHLCV values")
    prices = np.log(window[:, :4] / window[0, 3])
    volume = window[:, 4]
    relative_volume = np.log1p(volume / volume.mean()) if volume.mean() > 0 else volume
    return np.column_stack([prices, relative_volume]).astype(np.float32)


def iter_windows(
    sample_paths: list[Path],
    prepared: Path,
    *,
    context: int = 64,
    worker: int = 0,
    workers: int = 1,
    cursors: dict | None = None,
    decision_cutoff: str | None = None,
):
    """Cursor = siguiente fila confirmada por consumidor, no fila prefetched."""
    if context < 2:
        raise ValueError("Context must cover at least two sessions")
    cursors = cursors or {}
    cutoff = date.fromisoformat(decision_cutoff) if decision_cutoff else None
    for path in assigned(sorted(sample_paths), worker=worker, workers=workers):
        market, symbol = path.parent.parent.name, path.parent.name
        key = f"{market}/{symbol}"
        source_folder = prepared / market / symbol
        source_manifest = json.loads((source_folder / "manifest.json").read_text())
        sample_manifest = json.loads((path.parent / "manifest.json").read_text())
        if (
            source_manifest["fingerprint"] != sample_manifest["prepared_fingerprint"]
            or sha256(path) != sample_manifest["samples_sha256"]
            or sha256(source_folder / "prices.parquet")
            != source_manifest["artifacts"]["prices.parquet"]
        ):
            raise ValueError("Prepared or sample artifact changed since confirmation")
        cursor = cursors.get(key, 0)
        if type(cursor) is not int or cursor < 0:
            raise ValueError("Invalid confirmed cursor")
        with pq.ParquetFile(source_folder / "prices.parquet") as price_file:
            prices = price_file.read(
                columns=["open", "high", "low", "close", "volume"], use_threads=False
            )
        prices = np.column_stack([prices[name].to_numpy() for name in prices.column_names])
        offset = 0
        names = ("news", "charts", "fundamentals", "macro")
        columns = ["price_end_index", "prediction_at", *names]
        with pq.ParquetFile(path) as sample_file:
            for batch in sample_file.iter_batches(
                batch_size=256, columns=columns, use_threads=False
            ):
                ends = batch.column("price_end_index").to_pylist()
                timestamps = batch.column("prediction_at").to_pylist()
                vectors = {}
                for name in names:
                    column = batch.column(name)
                    lengths = pc.list_value_length(column).to_numpy()
                    if column.null_count or not len(lengths) or not (lengths == lengths[0]).all():
                        raise ValueError("Missing or inconsistent modality dimensions")
                    if not 1 <= lengths[0] <= 2048:
                        raise ValueError("Modality dimension exceeds the input contract")
                    vectors[name] = np.asarray(
                        column.flatten().to_numpy(), dtype=np.float32
                    ).reshape(len(batch), int(lengths[0]))
                for index, end in enumerate(ends):
                    offset += 1
                    if offset <= cursor:
                        continue
                    if cutoff and timestamps[index].date() > cutoff:
                        continue
                    if type(end) is not int or end < context - 1 or end >= len(prices):
                        raise ValueError("Window lies outside prepared price history")
                    inputs = {name: values[index].copy() for name, values in vectors.items()}
                    if any(not np.isfinite(v).all() for v in inputs.values()):
                        raise ValueError("Missing or nonfinite modality vector")
                    inputs["prices"] = price_features(prices[end - context + 1 : end + 1])
                    yield {
                        "inputs": inputs,
                        "cursor": (key, offset),
                        "prediction_at": timestamps[index],
                    }
        if cursor > offset:
            raise ValueError("Confirmed cursor exceeds artifact length")
