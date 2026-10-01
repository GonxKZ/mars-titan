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
from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.cohort_news import COHORT_POLICIES
from mars_titan.data.residual_arrays import residual_targets_array
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.temporal import MarketClock

from .cohort_contract import (
    cohort_identity,
    representation_hash,
    representation_identity,
    validate_cohort_rows,
)
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


def _asset_sources(asset, prepared, samples, context, cohort=None):
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
        item.get("news_content_policy")
        != (COHORT_POLICIES[cohort] if cohort else "verified_full_articles")
        for item in (origin, representation)
    ):
        raise ValueError("El entrenamiento requiere noticias completas verificadas")
    if cohort is not None and any(
        item.get("cohort_id") != cohort for item in (origin, representation)
    ):
        raise ValueError("Las noticias preparadas y sus vectores no pertenecen a la misma cohorte")
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
    return prices, vectors, hashes, representation_identity(representation) if cohort else None


def _label_batches(path, calculated, audit, cohort=None):
    schema = (
        LABEL_SCHEMA if cohort is None else LABEL_SCHEMA.append(pa.field("cohort_id", pa.string()))
    )
    by_time = {row.prediction_at: row for row in calculated.itertuples()}
    position, pending = 0, []
    with pq.ParquetFile(path) as file:
        if file.metadata.num_rows == 0:
            yield pa.Table.from_pylist([], schema=schema)
            return
        for batch in file.iter_batches(
            batch_size=1024,
            columns=["prediction_at"] + (["cohort_id"] if cohort else []),
            use_threads=False,
        ):
            validate_cohort_rows(pa.Table.from_batches([batch]), cohort)
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
                        **({"cohort_id": cohort} if cohort else {}),
                    )
                )
                position += 1
            if pending:
                yield pa.Table.from_pylist(pending, schema=schema)
                pending = []


def prepare_corpus_targets(
    manifest: Path, prepared: Path, output: Path, *, backend: str = "numpy"
) -> dict:
    """Confirmar etiquetas por activo. Una interrupción no publica un corpus parcial."""
    implementations = {"reference": residual_targets, "numpy": residual_targets_array}
    if backend not in implementations:
        raise ValueError("El motor de etiquetas debe ser reference o numpy")
    if output.is_symlink():
        raise ValueError("El directorio de salida no puede ser un enlace")
    meta, manifest_hash = read_manifest(manifest, 8 * 1024**2)
    cohort = cohort_identity(meta)
    if (
        meta.get("schema_version") not in {1, 2}
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
        "backend": backend,
        "source_manifest_sha256": manifest_hash,
        "prepared_root": str(prepared),
        "code_sha256": sha256(Path(__file__)),
        "target_reference_sha256": sha256(Path(__file__).parents[1] / "data/budget_targets.py"),
        "target_implementation_sha256": sha256(
            Path(__file__).parents[1] / "data/residual_arrays.py"
        ),
        "temporal_reference_sha256": sha256(Path(__file__).parents[1] / "data/temporal.py"),
        "pyarrow_version": pa.__version__,
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "exchange_calendars_version": exchange_calendars.__version__,
        "cohort_contract_sha256": sha256(Path(__file__).with_name("cohort_contract.py")),
        "cohort_files_sha256": sha256(Path(__file__).parents[1] / "data/cohort_files.py"),
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
    common_representation = None
    clocks = {m: MarketClock(m, meta["calendar_start"][m], "2024-01-05") for m in factors}
    for asset in meta["assets"]:
        market, symbol = asset["market"], asset["symbol"]
        prices, vectors, hashes, representation = _asset_sources(
            asset, prepared, samples, meta["context_sessions"], cohort
        )
        if cohort:
            if common_representation is not None and common_representation != representation:
                raise ValueError("No se pueden combinar representaciones con distinta semántica")
            common_representation = representation
        expected_receipt = dict(
            market=market,
            symbol=symbol,
            prices_sha256=hashes["prices"],
            samples_sha256=hashes["samples"],
        )
        if cohort:
            expected_receipt.update(
                cohort_id=cohort, representation_sha256=representation_hash(representation)
            )
        fingerprint = hashlib.sha256(
            json.dumps([configuration, hashes], sort_keys=True).encode()
        ).hexdigest()
        destination = output / "labels" / market / symbol
        label_path, receipt_path = destination / "labels.parquet", destination / "receipt.json"
        receipt = _json(receipt_path) if receipt_path.exists() else None
        if receipt and receipt["fingerprint"] != fingerprint:
            raise ValueError("Han cambiado los artefactos de la edición ya iniciada")
        if receipt and any(receipt.get(k) != v for k, v in expected_receipt.items()):
            raise ValueError("El recibo no corresponde a la identidad y cohorte esperadas")
        if receipt and label_path.is_file() and sha256(label_path) == receipt["labels_sha256"]:
            observed = read_bounded_table(label_path, max_rows=1_000_000)
            validate_cohort_rows(observed, cohort)
            partitions = Counter(observed["partition"].to_pylist())
            if receipt.get("samples") != len(observed) or receipt.get("counts") != {
                p: partitions[p] for p in ("train", "validation")
            }:
                raise ValueError("Los recuentos del recibo no coinciden con las etiquetas")
            reused += 1
        else:
            price_frame = read_bounded_table(prices, max_rows=200_000).to_pandas()
            calculated = implementations[backend](
                price_frame, factors[market], clocks[market], cutoff="2023-12-31"
            )
            audit = Counter()
            total = atomic_parquet_batches(
                label_path, _label_batches(vectors, calculated, audit, cohort)
            )
            if sha256(prices) != hashes["prices"] or sha256(vectors) != hashes["samples"]:
                raise ValueError("Las entradas cambiaron durante la preparación de etiquetas")
            receipt = dict(
                fingerprint=fingerprint,
                **expected_receipt,
                samples=total,
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
    if sha256(manifest) != manifest_hash:
        raise ValueError("La edición materializada cambió durante la supervisión")
    result = {
        "schema_version": meta["schema_version"],
        **({"cohort_id": cohort, "news_content_policy": COHORT_POLICIES[cohort]} if cohort else {}),
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
    if cohort:
        result.update(
            {k: meta[k] for k in ("coverage", "candidate_count", "samples", "failed_assets")}
        )
        result["representation"] = common_representation
        for key in ("markets", "preparation_scope", "parent_preparation"):
            if key in meta:
                result[key] = meta[key]
        cohort_identity(result)
    atomic_json(output / "manifest.json", result)
    return {**result, "reused_assets": reused}
