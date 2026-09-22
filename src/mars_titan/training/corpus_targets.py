"""Etiquetas y exclusiones por activo, con confirmación y población reconciliable."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import exchange_calendars
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.batches import atomic_parquet_batches, read_bounded_table
from mars_titan.data.budget_targets import residual_targets
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.temporal import MarketClock

from .corpus_inputs import _unique

LABEL_SCHEMA = pa.schema(
    [
        ("sample_row", pa.int64()),
        ("prediction_at", pa.timestamp("us", tz="UTC")),
        ("target_available_at", pa.timestamp("us", tz="UTC")),
        ("target", pa.float64()),
        ("partition", pa.string()),
        ("reason", pa.string()),
    ]
)


def _json(path, maximum=8 * 1024**2):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError("Falta un manifiesto regular y acotado")
    return json.loads(path.read_text(), object_pairs_hook=_unique)


def _path(root, market, symbol, name):
    path = root / market / symbol / name
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("Un artefacto no pertenece al origen o no es regular")
    return path


def _asset_sources(asset, prepared, samples, context):
    market, symbol = asset["market"], asset["symbol"]
    if (
        market not in {"US", "CN"}
        or not isinstance(symbol, str)
        or symbol in {".", ".."}
        or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", symbol)
    ):
        raise ValueError("La identidad de un activo no es válida")
    original = _path(prepared, market, symbol, "manifest.json")
    encoded = _path(samples, market, symbol, "manifest.json")
    origin = _json(original, 64 * 1024**2)
    representation = _json(encoded)
    if any(
        item.get("news_content_policy") != "verified_full_articles"
        for item in (origin, representation)
    ):
        raise ValueError("El entrenamiento requiere noticias completas verificadas")
    if (
        any(
            item.get("market") != market or item.get("symbol") != symbol
            for item in (origin, representation)
        )
        or representation.get("prepared_fingerprint") != origin.get("fingerprint")
        or representation.get("context_sessions") != context
    ):
        raise ValueError("La representación no corresponde al activo y contexto preparados")
    prices = _path(prepared, market, symbol, "prices.parquet")
    vectors = _path(samples, market, symbol, "samples.parquet")
    hashes = dict(
        prices=sha256(prices),
        samples=sha256(vectors),
        prepared_manifest=sha256(original),
        sample_manifest=sha256(encoded),
    )
    if (
        hashes["prices"] != origin["artifacts"]["prices.parquet"]
        or hashes["samples"] != representation["samples_sha256"]
    ):
        raise ValueError("Han cambiado las muestras o los precios preparados")
    return prices, vectors, hashes


def _label_batches(path, calculated, audit):
    by_time = {row.prediction_at: row for row in calculated.itertuples()}
    position, pending = 0, []
    with pq.ParquetFile(path) as file:
        if file.metadata.num_rows == 0:
            yield pa.Table.from_pylist([], schema=LABEL_SCHEMA)
            return
        for batch in file.iter_batches(
            batch_size=1024, columns=["prediction_at"], use_threads=False
        ):
            for moment in batch.column(0).to_pylist():
                if moment is None or moment.tzinfo is None:
                    raise ValueError("La muestra necesita una fecha con zona")
                label = by_time.get(moment) if moment.year <= 2023 else None
                reason = (
                    label.reason
                    if label is not None
                    else ("final_test_reserved" if moment.year > 2023 else "outside_label_calendar")
                )
                target, maturity, partition = None, None, None
                if reason == "accepted":
                    maturity, target = label.target_available_at, label.target
                    if moment.year <= 2022 and maturity.year <= 2022:
                        partition = "train"
                    elif moment.year == maturity.year == 2023:
                        partition = "validation"
                    else:
                        reason = "target_crosses_partition_boundary"
                audit[partition or reason] += 1
                pending.append(
                    dict(
                        sample_row=position,
                        prediction_at=moment,
                        target_available_at=maturity,
                        target=target,
                        partition=partition,
                        reason=reason,
                    )
                )
                position += 1
            if pending:
                yield pa.Table.from_pylist(pending, schema=LABEL_SCHEMA)
                pending = []


def prepare_corpus_targets(manifest: Path, prepared: Path, output: Path) -> dict:
    """Confirmar etiquetas por activo. Una interrupción no publica un corpus parcial."""
    if output.is_symlink():
        raise ValueError("El directorio de salida no puede ser un enlace")
    meta = _json(manifest)
    if (
        meta.get("schema_version") != 1
        or meta.get("kind") != "materialized_corpus"
        or meta.get("scope") not in {"development_snapshot", "full_corpus"}
        or type(meta.get("cohort_complete")) is not bool
        or (meta["scope"] == "full_corpus" and not meta["cohort_complete"])
        or type(meta.get("context_sessions")) is not int
        or not 2 <= meta["context_sessions"] <= 512
        or not isinstance(meta.get("assets"), list)
        or not meta["assets"]
    ):
        raise ValueError("La edición materializada no cumple el contrato")
    samples = Path(meta["samples_root"]).resolve()
    prepared, output = prepared.resolve(), output.resolve()
    for source in (prepared, samples, Path("dataset").resolve()):
        outside_source(source, output)
        outside_source(output, source)
    if len({(a["market"], a["symbol"]) for a in meta["assets"]}) != len(meta["assets"]):
        raise ValueError("Hay activos duplicados en el corpus")
    factors = {}
    for market in {a["market"] for a in meta["assets"]}:
        specification = meta["market_factors"].get(market)
        if not specification or specification.get("market") != market:
            raise ValueError("Falta un factor del mercado correspondiente")
        path = Path(specification["prices_path"])
        outside_source(output, path)
        if (
            not path.is_file()
            or path.is_symlink()
            or sha256(path) != specification["prices_sha256"]
        ):
            raise ValueError("El factor de mercado ha cambiado o no tiene una fuente regular")
        factors[market] = read_bounded_table(path, max_rows=200_000).to_pandas()
    configuration = {
        "source_manifest_sha256": sha256(manifest),
        "prepared_root": str(prepared),
        "code_sha256": sha256(Path(__file__)),
        "target_reference_sha256": sha256(Path(__file__).parents[1] / "data/budget_targets.py"),
        "temporal_reference_sha256": sha256(Path(__file__).parents[1] / "data/temporal.py"),
        "pyarrow_version": pa.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "exchange_calendars_version": exchange_calendars.__version__,
    }
    identity_path = output / "configuration.json"
    if output.exists() and not identity_path.exists() and next(output.iterdir(), None) is not None:
        raise ValueError("El directorio de salida contiene datos de otra ejecución")
    if identity_path.exists() and _json(identity_path) != configuration:
        raise ValueError("La configuración pertenece a otra edición de datos o código")
    output.mkdir(parents=True, exist_ok=True)
    if not identity_path.exists():
        atomic_json(identity_path, configuration)
    assets, reused, counts = [], 0, Counter(train=0, validation=0)
    clocks = {m: MarketClock(m, meta["calendar_start"][m], "2024-01-05") for m in factors}
    for asset in meta["assets"]:
        market, symbol = asset["market"], asset["symbol"]
        prices, vectors, hashes = _asset_sources(asset, prepared, samples, meta["context_sessions"])
        fingerprint = hashlib.sha256(
            json.dumps([configuration, hashes], sort_keys=True).encode()
        ).hexdigest()
        destination = output / "labels" / market / symbol
        label_path, receipt_path = destination / "labels.parquet", destination / "receipt.json"
        receipt = _json(receipt_path) if receipt_path.exists() else None
        if receipt and receipt["fingerprint"] != fingerprint:
            raise ValueError("Han cambiado los artefactos de la edición ya iniciada")
        if receipt and label_path.is_file() and sha256(label_path) == receipt["labels_sha256"]:
            reused += 1
        else:
            price_frame = read_bounded_table(prices, max_rows=200_000).to_pandas()
            calculated = residual_targets(
                price_frame, factors[market], clocks[market], cutoff="2023-12-31"
            )
            audit = Counter()
            total = atomic_parquet_batches(label_path, _label_batches(vectors, calculated, audit))
            if sha256(prices) != hashes["prices"] or sha256(vectors) != hashes["samples"]:
                raise ValueError("Las entradas cambiaron durante la preparación de etiquetas")
            receipt = dict(
                fingerprint=fingerprint,
                market=market,
                symbol=symbol,
                samples=total,
                prices_sha256=hashes["prices"],
                samples_sha256=hashes["samples"],
                labels_sha256=sha256(label_path),
                counts={p: audit[p] for p in ("train", "validation")},
                excluded_reasons={
                    k: v for k, v in sorted(audit.items()) if k not in {"train", "validation"}
                },
                calendar_label_reasons={
                    str(k): int(v) for k, v in calculated.reason.value_counts().items()
                },
            )
            atomic_json(receipt_path, receipt)
        assets.append(receipt)
        counts.update(receipt["counts"])
    result = {
        "schema_version": 1,
        "kind": "corpus_supervision",
        "scope": meta["scope"],
        "cohort_complete": meta["cohort_complete"],
        "context_sessions": meta["context_sessions"],
        "roots": {
            "prepared": str(prepared),
            "samples": str(samples),
            "labels": str(output / "labels"),
        },
        "assets": assets,
        "counts": dict(counts),
        "configuration": configuration,
        "market_factors": meta["market_factors"],
        "final_test_opened": False,
    }
    atomic_json(output / "manifest.json", result)
    return {**result, "reused_assets": reused}
