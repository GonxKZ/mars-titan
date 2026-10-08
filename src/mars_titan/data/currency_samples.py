"""Edición USD/CAD con proyección de vectores existentes y objetivos identificados."""

import copy
import fcntl
import re
import resource
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.training.cohort_contract import cohort_identity, representation_hash
from mars_titan.training.corpus_targets import _label_batches

from .accounting_catalog import CAD_CONCEPTS, COMMON_CONCEPTS
from .accounting_catalog import USD_CONCEPTS as USD_CONCEPTS
from .batches import atomic_parquet_batches, read_bounded_table
from .cohort_contexts import MacroVectors
from .cohort_files import read_manifest, safe_destination
from .cohort_samples import _digest, _schema, materialize_cohort_asset
from .embeddings import EmbeddingCache
from .residual_arrays import residual_targets_array
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

POLICY = "separate_usd_cad_channels_v1"
_LIMIT = 64 * 1024**2
_SEAL = np.datetime64("2024-01-01", "us").astype(np.int64)


def _checked(path, expected=None):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Falta un archivo regular del origen")
    actual = sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"Ha cambiado la huella de origen: {path}")
    return actual


def _signature(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _times(column):
    if not pa.types.is_timestamp(column.type) or column.type.tz != "UTC" or column.null_count:
        raise ValueError("Las decisiones deben ser instantes UTC completos")
    values = pc.cast(column, pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy()
    if np.any(values >= _SEAL) or np.any(values[1:] <= values[:-1]):
        raise ValueError("Las decisiones deben estar ordenadas y mantener cerrado 2024")
    return values


def _metadata(file, column="prediction_at"):
    if file.metadata.num_rows > 1_000_000 or any(
        file.metadata.row_group(i).num_rows > 100_000
        or file.metadata.row_group(i).total_byte_size > _LIMIT
        for i in range(file.num_row_groups)
    ):
        raise ValueError("El archivo supera el presupuesto por activo o grupo")
    table = file.read(columns=[column], use_threads=False)
    if table.nbytes > _LIMIT:
        raise ValueError("Los metadatos superan el presupuesto")
    return table[column]


def _macro_inputs(macro_path, admission_path, catalog_path, expected_ids, *, market="US"):
    report, receipt_hash = read_manifest(admission_path, 8 * 1024**2)
    if (
        market not in {"US", "CN"}
        or report.get("market") != market
        or report.get("required_indicator_count") != 140
        or report.get("required_indicator_ids") != expected_ids
        or report.get("catalog_sha256") != _checked(catalog_path)
        or report.get("source_sha256") != _checked(macro_path)
        or report.get("complete_decisions_path") != "complete-decisions.parquet"
    ):
        raise ValueError("La admisión no corresponde a los 140 indicadores de la edición")
    path = admission_path.parent / "complete-decisions.parquet"
    _checked(path, report["complete_decisions_sha256"])
    table = read_bounded_table(path, max_rows=20_000)
    times = _times(table["prediction_at"])
    if len(times) != report.get("complete_decisions") or not len(times):
        raise ValueError("El recuento de decisiones macro no concilia")
    if not np.all(table["indicator_count"].to_numpy() == 140):
        raise ValueError("Una decisión no contiene todos los indicadores")
    with pq.ParquetFile(macro_path) as file:
        moments = _metadata(file)
        if np.any(
            pc.cast(moments, pa.timestamp("us", tz="UTC")).cast(pa.int64()).to_numpy() >= _SEAL
        ):
            raise ValueError("El panel macro invade la reserva final")
    macros = MacroVectors(macro_path)
    admitted = frozenset(table["prediction_at"].to_pylist())
    for moment in admitted:
        value = macros.at(moment)
        if (
            value is None
            or value[1] > moment
            or not np.isfinite(value[0]).all()
            or not np.all(value[0][140:280] == 1)
        ):
            raise ValueError("La decisión admitida no tiene 140 valores disponibles")
    return macros, admitted, receipt_hash, report["complete_decisions_sha256"]


class _Cache:
    """Escribir solo en la caché nueva y contrastar los vectores reutilizados."""

    def __init__(self, path, source):
        self.current = EmbeddingCache(path)
        self.source = EmbeddingCache(source, read_only=True) if source else None

    def get(self, identity):
        value = self.current.get(identity)
        return self.source.get(identity) if value is None and self.source else value

    def put(self, identity, vector):
        self.current.put(identity, vector)

    def close(self):
        self.current.close()
        if self.source:
            self.source.close()


def _cad_candidates(parent, prepared):
    review, selected = [], set()
    for asset in parent["coverage"]:
        if asset["state"] != "encoded" or asset["samples"]:
            continue
        symbol = asset["symbol"]
        folder = prepared / "US" / symbol
        receipt, digest = read_manifest(folder / "manifest.json")
        count = receipt["counts"]["fundamentals"]
        found = False
        if count:
            path = folder / "fundamentals.parquet"
            _checked(path, receipt["artifacts"]["fundamentals.parquet"])
            with pq.ParquetFile(path) as file:
                if file.metadata.num_rows > 200_000:
                    raise ValueError("El inventario contable supera el presupuesto")
                for batch in file.iter_batches(
                    batch_size=1024, columns=["concept", "available_at"]
                ):
                    if any(t.year >= 2024 for t in batch["available_at"].to_pylist()):
                        raise ValueError("Los hechos preparados invaden la reserva final")
                    found |= bool(set(batch["concept"].to_pylist()) & set(CAD_CONCEPTS))
        if found:
            selected.add(symbol)
        review.append(
            dict(
                symbol=symbol,
                prepared_sha256=digest,
                cad_candidate=found,
                reason="cad_balance_present" if found else "no_prepared_cad_balance",
            )
        )
    return selected, review


def _project_samples(source, destination, macros, admitted, *, legacy, batch_rows):
    """Copiar solo decisiones admitidas, conservando las columnas no transformadas."""
    selected_rows = []
    schema = _schema(COMMON_CONCEPTS, macros.indicators).append(
        pa.field("source_sample_row", pa.int64())
    )
    with pq.ParquetFile(source) as file:
        moments = _metadata(file)
        times = _times(moments)
        accepted = np.array([int(t.timestamp() * 1_000_000) for t in admitted], dtype=np.int64)
        selected = np.flatnonzero(np.isin(times, accepted))
        original_rows = len(times)

        def batches():
            offset, emitted = 0, False
            columns = [
                name
                for name in schema.names
                if name not in {"macro", "macro_available_at", "source_sample_row"}
            ]
            for group in range(file.num_row_groups):
                count = file.metadata.row_group(group).num_rows
                chosen = selected[(selected >= offset) & (selected < offset + count)] - offset
                if not len(chosen):
                    offset += count
                    continue
                last = int(chosen[-1]) + 1
                position = 0
                for chunk in file.iter_batches(
                    batch_size=batch_rows, row_groups=[group], columns=columns, use_threads=False
                ):
                    if position >= last:
                        break
                    local = (
                        chosen[(chosen >= position) & (chosen < position + len(chunk))] - position
                    )
                    if len(local):
                        table = pa.Table.from_batches([chunk.take(pa.array(local))])
                        if table.nbytes > _LIMIT:
                            raise ValueError("El lote de muestras supera 64 MiB")
                        for name, width in (
                            ("news", 384),
                            ("charts", 512),
                            ("fundamentals", 45 if legacy else 69),
                        ):
                            column = table[name].combine_chunks()
                            if (
                                column.null_count
                                or not np.all(pc.list_value_length(column).to_numpy() == width)
                                or not np.isfinite(column.flatten().to_numpy()).all()
                            ):
                                raise ValueError("Falta una modalidad o su dimensión no es válida")
                        if legacy:
                            previous = (
                                table["fundamentals"]
                                .combine_chunks()
                                .flatten()
                                .to_numpy()
                                .reshape(-1, 3, 15)
                            )
                            values = np.zeros((len(table), 3, 23), dtype=np.float32)
                            values[:, :, :15] = previous
                            column = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel()), 69)
                            table = table.set_column(
                                table.schema.get_field_index("fundamentals"), "fundamentals", column
                            )
                        dates = table["prediction_at"].to_pylist()
                        contexts = [macros.at(t) for t in dates]
                        available = table["input_availability"].to_pylist()
                        for inputs, moment, value in zip(available, dates, contexts, strict=True):
                            inputs["macro"] = value[1]
                            if any(v is None or v > moment for v in inputs.values()):
                                raise ValueError("Una modalidad no acredita disponibilidad causal")
                        table = table.set_column(
                            table.schema.get_field_index("input_availability"),
                            "input_availability",
                            pa.array(available, type=schema.field("input_availability").type),
                        )
                        table = table.append_column(
                            "macro",
                            pa.FixedSizeListArray.from_arrays(
                                pa.array(np.stack([v[0] for v in contexts]).ravel()), 420
                            ),
                        )
                        table = table.append_column(
                            "macro_available_at",
                            pa.array(
                                [v[1] for v in contexts],
                                type=schema.field("macro_available_at").type,
                            ),
                        )
                        indexes = offset + position + local
                        table = table.append_column(
                            "source_sample_row", pa.array(indexes, type=pa.int64())
                        )
                        selected_rows.extend(indexes.tolist())
                        emitted = True
                        yield table.select(schema.names).cast(schema)
                    position += len(chunk)
                offset += count
            if not emitted:
                yield pa.Table.from_pylist([], schema=schema)

        count = atomic_parquet_batches(destination, batches())
    return np.asarray(selected_rows, dtype=np.int64), original_rows, count


def _remap_labels(source, destination, indices, samples_path, *, batch_rows):
    with pq.ParquetFile(source) as file:
        _times(_metadata(file))
    labels = read_bounded_table(source, max_rows=1_000_000)
    positions = labels["sample_row"].to_numpy()
    if (
        len(np.unique(positions)) != len(positions)
        or np.any(positions < 0)
        or np.any(positions >= len(positions))
    ):
        raise ValueError("Las etiquetas de origen no identifican cada muestra")
    lookup = np.empty(len(positions), dtype=np.int64)
    lookup[positions] = np.arange(len(positions))
    result = labels.take(pa.array(lookup[indices]))
    with pq.ParquetFile(samples_path) as file:
        prediction = _metadata(file)
    if result["prediction_at"].to_pylist() != prediction.to_pylist():
        raise ValueError("La etiqueta y su muestra de origen no comparten decisión")
    result = result.set_column(
        result.schema.get_field_index("sample_row"),
        "sample_row",
        pa.array(np.arange(len(indices)), type=pa.int64()),
    )
    result = result.append_column("source_sample_row", pa.array(indices, type=pa.int64()))
    atomic_parquet_batches(
        destination,
        (pa.Table.from_batches([b]) for b in result.to_batches(max_chunksize=batch_rows))
        if len(result)
        else (result,),
    )


def _past_prices(path):
    parts = []
    with pq.ParquetFile(path) as file:
        sessions = _metadata(file, "session").to_pylist()
        if sessions != sorted(set(sessions)):
            raise ValueError("Los precios no están ordenados por sesión")
        offset = 0
        for group in range(file.num_row_groups):
            count = file.metadata.row_group(group).num_rows
            selected = [
                i for i, s in enumerate(sessions[offset : offset + count]) if s < "2024-01-01"
            ]
            if selected:
                last = selected[-1] + 1
                batch = next(
                    file.iter_batches(batch_size=last, row_groups=[group], use_threads=False)
                )
                if batch.nbytes > _LIMIT:
                    raise ValueError("Los precios superan el presupuesto de lectura")
                parts.append(pa.Table.from_batches([batch]))
            offset += count
    if not parts:
        raise ValueError("Faltan precios anteriores a 2024")
    return pa.concat_tables(parts).to_pandas()


def _label_receipt(path, samples, *, market, symbol, prices_hash, fingerprint, representation):
    table = read_bounded_table(path, max_rows=1_000_000)
    if len(table) != samples:
        raise ValueError("Las etiquetas no concilian con las muestras")
    counts = Counter(table["partition"].to_pylist())
    excluded = Counter(
        reason
        for part, reason in zip(
            table["partition"].to_pylist(), table["reason"].to_pylist(), strict=True
        )
        if part is None
    )
    return dict(
        fingerprint=fingerprint,
        market=market,
        symbol=symbol,
        prices_sha256=prices_hash,
        labels_sha256=sha256(path),
        samples=samples,
        counts={p: counts[p] for p in ("train", "validation")},
        excluded_reasons=dict(excluded),
        cohort_id="original_audited",
        representation_sha256=representation_hash(representation),
    )


def prepare_currency_corpus(
    parent_manifest,
    output,
    *,
    macro_path,
    admission_path,
    catalog_path,
    cad_assets=None,
    encoders=None,
    cache_path=None,
    source_cache=None,
    batch_rows=128,
    stop_after_assets=None,
    clock=None,
):
    """Confirmar activos recuperables y publicar ambos manifiestos solo al completar la edición."""
    began = time.monotonic()
    if type(batch_rows) is not int or not 1 <= batch_rows <= 512:
        raise ValueError("El lote debe contener entre una y 512 muestras")
    if stop_after_assets is not None and (
        type(stop_after_assets) is not int or stop_after_assets < 1
    ):
        raise ValueError("La pausa operativa debe ser un número positivo de activos")
    parent_manifest, output, macro_path, admission_path, catalog_path = map(
        Path, (parent_manifest, output, macro_path, admission_path, catalog_path)
    )
    safe_destination(output)
    cache_path = Path(cache_path) if cache_path else output / "embeddings.sqlite"
    source_cache = Path(source_cache) if source_cache else None
    safe_destination(cache_path)
    if not cache_path.resolve().is_relative_to(output.resolve()):
        raise ValueError("La caché nueva debe estar dentro del destino de la edición")
    if source_cache:
        outside_source(output, source_cache)
        if source_cache.is_symlink() or not source_cache.is_file():
            raise ValueError("La caché de origen debe ser un archivo regular")
    parent, parent_hash = read_manifest(parent_manifest, 8 * 1024**2)
    if (
        cohort_identity(parent) != "original_audited"
        or parent.get("kind") != "corpus_supervision"
        or parent.get("scope") != "full_corpus"
        or parent.get("cohort_complete") is not True
        or parent.get("final_test_opened") is not False
        or parent.get("markets") != ["US"]
        or "temporal_view" in parent
    ):
        raise ValueError("Se necesita la supervisión original US con la reserva cerrada")
    if any(
        a.get("market") != "US"
        or not isinstance(a.get("symbol"), str)
        or a["symbol"] in {".", ".."}
        or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", a["symbol"])
        for a in parent["coverage"]
    ):
        raise ValueError("La identidad de un símbolo no pertenece al universo US")
    if parent["representation"]["context_sessions"] != parent["context_sessions"]:
        raise ValueError("El contexto no corresponde a la representación original")
    if parent["representation"]["fundamental_concepts"] != list(USD_CONCEPTS):
        raise ValueError("La proyección requiere los quince conceptos USD originales")
    expected_encoder = parent["representation"]["encoders"]
    if encoders is not None and encoders.spec != expected_encoder:
        raise ValueError(
            "El encoder no conserva exactamente el contrato de los vectores originales"
        )
    roots = {k: Path(v) for k, v in parent["roots"].items()}
    for source in (
        *roots.values(),
        parent_manifest,
        macro_path,
        admission_path,
        catalog_path,
        Path("dataset"),
    ):
        outside_source(source, output)
        outside_source(output, source)
    macros, admitted, admission_hash, decisions_hash = _macro_inputs(
        macro_path, admission_path, catalog_path, parent["representation"]["macro_indicators"]
    )
    candidates, review = _cad_candidates(parent, roots["prepared"])
    if cad_assets is not None and set(cad_assets) != candidates:
        raise ValueError("La selección CAD no cubre todos los candidatos del inventario")
    code = {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "currency_samples.py",
            "company_factors.py",
            "cohort_samples.py",
            "cohort_contexts.py",
            "samples.py",
            "batches.py",
            "storage.py",
            "residual_arrays.py",
        )
    }
    representation = copy.deepcopy(parent["representation"])
    representation["fundamental_concepts"] = list(COMMON_CONCEPTS)
    representation["representation_code"] = {**representation["representation_code"], **code}
    clock = clock or MarketClock("US", "1990-01-01", "2026-01-01")
    identity = dict(
        schema_version=1,
        policy=POLICY,
        parent_manifest_sha256=parent_hash,
        macro_sha256=macros.sha256,
        admission_sha256=admission_hash,
        decisions_sha256=decisions_hash,
        catalog_sha256=sha256(catalog_path),
        source_units={name: "CAD" for name in sorted(candidates)},
        encoders=expected_encoder,
        batch_rows=batch_rows,
        representation_sha256=representation_hash(representation),
        code_sha256=code,
        calendar_sha256=_digest([t.isoformat() for t in clock.decisions]),
        sources={
            str(parent_manifest): parent_hash,
            str(macro_path): macros.sha256,
            str(admission_path): admission_hash,
            str(catalog_path): sha256(catalog_path),
            str(admission_path.parent / "complete-decisions.parquet"): decisions_hash,
        },
    )
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".edition.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        config = output / "configuration.json"
        if config.exists():
            if read_manifest(config)[0] != identity:
                raise ValueError(
                    "La configuración pertenece a otra edición o ha cambiado una fuente"
                )
        elif any(p.name != ".edition.lock" for p in output.iterdir()):
            raise ValueError("El destino contiene una edición no identificada")
        else:
            atomic_json(config, identity)
        return _materialize(
            parent,
            parent_hash,
            roots,
            output,
            identity,
            representation,
            macros,
            admitted,
            candidates,
            review,
            encoders,
            cache_path,
            source_cache,
            batch_rows,
            stop_after_assets,
            clock,
            began,
        )


def _materialize(
    parent,
    parent_hash,
    roots,
    output,
    identity,
    representation,
    macros,
    admitted,
    candidates,
    review,
    encoders,
    cache_path,
    source_cache,
    batch_rows,
    stop_after_assets,
    clock,
    began,
):
    assets_by_symbol = {a["symbol"]: a for a in parent["assets"]}
    encoded_root, labels_root = output / "encoded/samples", output / "supervised/labels"
    coverage, assets, pending = [], [], []
    completed = reused = 0
    cache = None
    factor = None
    try:
        for original in parent["coverage"]:
            symbol = original["symbol"]
            if (
                original["state"] != "encoded"
                or not original["samples"]
                and symbol not in candidates
            ):
                coverage.append(dict(original))
                continue
            if symbol in candidates and encoders is None:
                pending.append(symbol)
                continue
            if stop_after_assets is not None and completed >= stop_after_assets:
                pending.extend(
                    a["symbol"]
                    for a in parent["coverage"]
                    if a["state"] == "encoded"
                    and (a["samples"] or a["symbol"] in candidates)
                    and a["symbol"] not in {x["symbol"] for x in coverage}
                    and a["symbol"] not in pending
                )
                break
            folder = encoded_root / "US" / symbol
            labels_folder = labels_root / "US" / symbol
            safe_destination(folder)
            safe_destination(labels_folder)
            folder.mkdir(parents=True, exist_ok=True)
            labels_folder.mkdir(parents=True, exist_ok=True)
            origin, origin_hash = read_manifest(roots["prepared"] / "US" / symbol / "manifest.json")
            prices = roots["prepared"] / "US" / symbol / "prices.parquet"
            prices_hash = _checked(prices, origin["artifacts"]["prices.parquet"])
            source_unit = "CAD" if symbol in candidates else "USD"
            source_manifest_hash = None
            guarded = {prices: _signature(prices)}
            if source_unit == "USD":
                old_path = roots["samples"] / "US" / symbol / "manifest.json"
                old_receipt, source_manifest_hash = read_manifest(old_path)
                if (
                    representation_hash(old_receipt)
                    != assets_by_symbol[symbol]["representation_sha256"]
                    or old_receipt.get("samples_sha256")
                    != assets_by_symbol[symbol]["samples_sha256"]
                    or old_receipt.get("prepared_fingerprint") != origin["fingerprint"]
                ):
                    raise ValueError(
                        "La representación de origen no acredita su encoder y sus datos"
                    )
                guarded[old_path] = _signature(old_path)
            fingerprint = _digest(
                [identity, symbol, origin_hash, source_unit, source_manifest_hash]
            )
            receipt_path = labels_folder / "receipt.json"
            sample_manifest = folder / "manifest.json"
            if receipt_path.exists() and sample_manifest.exists():
                receipt = read_manifest(receipt_path)[0]
                encoded = read_manifest(sample_manifest)[0]
                if (
                    receipt.get("fingerprint") != fingerprint
                    or encoded.get("fingerprint") != fingerprint
                ):
                    raise ValueError("El activo pertenece a otra identidad de edición")
                _checked(folder / "samples.parquet", encoded["samples_sha256"])
                _checked(labels_folder / "labels.parquet", receipt["labels_sha256"])
                expected_receipt = _label_receipt(
                    labels_folder / "labels.parquet",
                    encoded["samples"],
                    market="US",
                    symbol=symbol,
                    prices_hash=prices_hash,
                    fingerprint=fingerprint,
                    representation=representation,
                )
                expected_receipt["samples_sha256"] = encoded["samples_sha256"]
                if any(receipt.get(key) != value for key, value in expected_receipt.items()):
                    raise ValueError(
                        "El recibo no concilia con los recuentos y etiquetas verificados"
                    )
                if representation_hash(encoded) != identity["representation_sha256"]:
                    raise ValueError("La representación del recibo reutilizado ha cambiado")
                if source_unit == "USD":
                    _checked(
                        roots["samples"] / "US" / symbol / "samples.parquet",
                        assets_by_symbol[symbol]["samples_sha256"],
                    )
                    _checked(
                        roots["labels"] / "US" / symbol / "labels.parquet",
                        assets_by_symbol[symbol]["labels_sha256"],
                    )
                reused += 1
            else:
                if source_unit == "CAD":
                    if cache is None:
                        cache = _Cache(cache_path, source_cache)
                    raw = output / "cad-materialized/US" / symbol
                    raw_receipt = materialize_cohort_asset(
                        roots["prepared"] / "US" / symbol,
                        raw,
                        clock,
                        macros,
                        encoders,
                        cache,
                        cohort="original_audited",
                        context=parent["context_sessions"],
                        news_lookback_sessions=representation["news_lookback_sessions"],
                        fundamental_concepts=COMMON_CONCEPTS,
                        source_unit="CAD",
                        admitted_decisions=admitted,
                        batch_rows=batch_rows,
                    )
                    source = raw / "samples.parquet"
                    guarded[source] = _signature(source)
                    _checked(source, raw_receipt["samples_sha256"])
                    lineage = dict(
                        kind="new_cad_encoding"
                        if any(raw_receipt["cache_misses"].values())
                        else "cached_cad_materialization",
                        neural_inference_executed=any(raw_receipt["cache_misses"].values()),
                        source_samples_sha256=raw_receipt["samples_sha256"],
                        source_receipt_sha256=sha256(raw / "manifest.json"),
                        encoder=encoders.spec,
                    )
                else:
                    source = roots["samples"] / "US" / symbol / "samples.parquet"
                    guarded[source] = _signature(source)
                    original_asset = assets_by_symbol[symbol]
                    _checked(source, original_asset["samples_sha256"])
                    lineage = dict(
                        kind="usd_projection",
                        source_samples_sha256=original_asset["samples_sha256"],
                        source_labels_sha256=original_asset["labels_sha256"],
                        source_receipt_sha256=source_manifest_hash,
                        source_representation_sha256=original_asset["representation_sha256"],
                        encoder=representation["encoders"],
                    )
                indices, old_count, count = _project_samples(
                    source,
                    folder / "samples.parquet",
                    macros,
                    admitted,
                    legacy=source_unit == "USD",
                    batch_rows=batch_rows,
                )
                label_path = labels_folder / "labels.parquet"
                if source_unit == "USD":
                    original_labels = roots["labels"] / "US" / symbol / "labels.parquet"
                    guarded[original_labels] = _signature(original_labels)
                    _checked(original_labels, lineage["source_labels_sha256"])
                    _remap_labels(
                        original_labels,
                        label_path,
                        indices,
                        folder / "samples.parquet",
                        batch_rows=batch_rows,
                    )
                else:
                    factor_spec = parent["market_factors"]["US"]
                    if factor is None:
                        factor_path = Path(factor_spec["prices_path"])
                        _checked(factor_path, factor_spec["prices_sha256"])
                        factor = _past_prices(factor_path)
                    calculated = residual_targets_array(
                        _past_prices(prices), factor, clock, cutoff="2023-12-31"
                    )
                    audit = Counter()

                    def batches(folder=folder, calculated=calculated, audit=audit):
                        for table in _label_batches(
                            folder / "samples.parquet", calculated, audit, "original_audited"
                        ):
                            yield table.append_column("source_sample_row", table["sample_row"])

                    atomic_parquet_batches(label_path, batches())
                encoded = dict(
                    schema_version=3,
                    cohort_id="original_audited",
                    market="US",
                    symbol=symbol,
                    news_content_policy=parent["news_content_policy"],
                    training_ready=False,
                    fingerprint=fingerprint,
                    prepared_fingerprint=origin["fingerprint"],
                    **representation,
                    samples=count,
                    samples_sha256=sha256(folder / "samples.parquet"),
                    macro_sha256=macros.sha256,
                    source_unit=source_unit,
                    currency_projection=dict(
                        policy=POLICY, parent_manifest_sha256=parent_hash, source=lineage
                    ),
                    excluded_reasons=dict(outside_macro_admission=old_count - count),
                    source_samples=old_count,
                )
                if source_unit == "CAD":
                    encoded["materialization_exclusions"] = raw_receipt["excluded_reasons"]
                if any(_signature(path) != stat for path, stat in guarded.items()):
                    raise ValueError("Una fuente cambió durante la proyección")
                atomic_json(sample_manifest, encoded)
                receipt = _label_receipt(
                    label_path,
                    count,
                    market="US",
                    symbol=symbol,
                    prices_hash=prices_hash,
                    fingerprint=fingerprint,
                    representation=representation,
                )
                receipt.update(samples_sha256=encoded["samples_sha256"], source_lineage=lineage)
                atomic_json(receipt_path, receipt)
            coverage.append(
                dict(
                    market="US",
                    symbol=symbol,
                    state="encoded",
                    samples=encoded["samples"],
                    fingerprint=fingerprint,
                )
            )
            if encoded["samples"]:
                assets.append(receipt)
            completed += 1
            if completed % 32 == 0:
                atomic_json(
                    output / "progress.json",
                    dict(
                        status="materializing",
                        completed_assets=completed,
                        reused_assets=reused,
                        pending_cad=sorted(pending),
                        final_test_opened=False,
                    ),
                )
    finally:
        if cache is not None:
            cache.close()
    macros.verify()
    for path, digest in identity["sources"].items():
        _checked(Path(path), digest)
    if any(
        sha256(Path(__file__).with_name(name)) != digest
        for name, digest in identity["code_sha256"].items()
    ):
        raise ValueError("El código cambió durante la edición")
    result = dict(
        status="requires_encoders"
        if pending and encoders is None
        else "paused"
        if pending
        else "completed",
        completed_assets=completed,
        reused_assets=reused,
        pending_cad=sorted(pending),
        currency_review=review,
        final_test_opened=False,
        elapsed_seconds=time.monotonic() - began,
        process_lifetime_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    )
    if not pending:
        total = sum(a["samples"] for a in assets)
        shared = dict(
            schema_version=2,
            cohort_id="original_audited",
            news_content_policy=parent["news_content_policy"],
            scope="full_corpus",
            cohort_complete=True,
            context_sessions=parent["context_sessions"],
            markets=["US"],
            coverage=coverage,
            candidate_count=len(coverage),
            samples=total,
            failed_assets=0,
            market_factors=parent["market_factors"],
            final_test_opened=False,
            macro_catalog_sha256=identity["catalog_sha256"],
            currency_projection=identity,
        )
        encoded = dict(
            **shared,
            kind="materialized_corpus",
            samples_root=str(encoded_root.resolve()),
            calendar_start=dict(US=clock.days[0].isoformat()),
            configuration=identity,
            assets=[
                dict(market=a["market"], symbol=a["symbol"], cohort_id="original_audited")
                for a in assets
            ],
        )
        encoded_path = output / "encoded/manifest.json"
        atomic_json(encoded_path, encoded)
        counts = {p: sum(a["counts"][p] for a in assets) for p in ("train", "validation")}
        supervised = dict(
            **shared,
            kind="corpus_supervision",
            roots=dict(
                prepared=str(roots["prepared"].resolve()),
                samples=str(encoded_root.resolve()),
                labels=str(labels_root.resolve()),
            ),
            assets=assets,
            counts=counts,
            representation=representation,
            configuration=dict(
                backend="currency_projection_and_numpy",
                source_manifest_sha256=sha256(encoded_path),
                source_manifest=str(encoded_path.resolve()),
                parent_supervision_sha256=parent_hash,
            ),
        )
        cohort_identity(supervised)
        atomic_json(output / "supervised/manifest.json", supervised)
        result.update(
            encoded_manifest=str(encoded_path.resolve()),
            supervised_manifest=str((output / "supervised/manifest.json").resolve()),
            samples=total,
            counts=counts,
        )
    atomic_json(output / "progress.json", result)
    return result
