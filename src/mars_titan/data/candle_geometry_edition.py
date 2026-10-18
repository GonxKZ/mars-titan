"""Derivar otra representación de velas sin recodificar las modalidades restantes."""

import argparse
import fcntl
import json
import os
import re
import resource
import shutil
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.training.cohort_contract import cohort_identity

from . import candle_geometry_selection as selection_policy
from .batches import atomic_parquet_batches, read_bounded_table
from .candle_geometry import MODEL, WIDTH, geometry_chart, geometry_encoder_spec
from .cohort_files import read_manifest, safe_destination
from .cohort_samples import _artifact, _digest
from .input_policy import HISTORICAL_MASKED
from .macro_coverage import _publish_directory
from .modality_ablation import list_column
from .price_windows import (
    absent_positions,
    calendar_digest,
    check_price_window_contract,
    check_row_positions,
    present_slots,
)
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock, aware

MAX_BYTES = 16 * 1024**2
MAX_PARTITION_BYTES = 64 * 1024**2
MAX_ROWS = 200_000


def _input_configuration(configuration):
    # El contraste solo cambia imágenes. La caché y las huellas de código quedan registradas
    # por separado, mientras el contrato efectivo de texto y el de las entradas deben coincidir.
    result = {k: v for k, v in configuration.items() if k not in {"code", "cache_charts"}}
    result["encoders"] = {
        k: v
        for k, v in configuration["encoders"].items()
        if k not in {"image_model", "image_policy", "image_url", "image_weights_sha256"}
    }
    return result


def _compare_samples(paths):
    with pq.ParquetFile(paths[0]) as before, pq.ParquetFile(paths[1]) as after:
        if (
            not before.schema_arrow.equals(after.schema_arrow, check_metadata=True)
            or before.metadata.num_rows != after.metadata.num_rows
        ):
            raise ValueError("Las representaciones no conservan las mismas filas y columnas")
        for file in (before, after):
            if file.metadata.num_rows > 200_000 or any(
                file.metadata.row_group(i).total_byte_size > 64 * 1024**2
                for i in range(file.num_row_groups)
            ):
                raise ValueError("Las muestras superan el presupuesto de comparación")
        columns = [
            name for name in before.schema_arrow.names if name not in {"charts", "chart_hash"}
        ]
        streams = [
            file.iter_batches(batch_size=128, columns=columns, use_threads=False)
            for file in (before, after)
        ]
        count = 0
        for left, right in zip(*streams, strict=True):
            if max(left.nbytes, right.nbytes) > MAX_BYTES:
                raise ValueError("El lote decodificado supera el presupuesto de comparación")
            if not left.equals(right, check_metadata=True):
                raise ValueError("Han cambiado filas, orden o entradas distintas de las velas")
            count += left.num_rows
        return count


def compare_representations(previous, current, output=None):
    """Comparar las mismas muestras y su preparación, sin reconstruir ventanas sin máscara."""
    roots = (Path(previous), Path(current))
    if output is not None:
        output = Path(output)
        for root in roots:
            outside_source(root, output)
    configurations = [read_manifest(root / "configuration.json") for root in roots]
    if _input_configuration(configurations[0][0]) != _input_configuration(configurations[1][0]):
        raise ValueError("La comparación necesita la misma preparación y los mismos contratos")
    manifests = [read_manifest(root / "manifest.json") for root in roots]
    assets, expected_samples = selection_policy.comparison_population(manifests, configurations)
    count = 0
    for market, symbol in assets:
        folders = [root / "samples" / market / symbol for root in roots]
        for root, folder in zip(roots, folders, strict=True):
            safe_destination(folder)
            if not folder.resolve().is_relative_to(root.resolve()):
                raise ValueError("Un activo queda fuera de la edición")
        receipts = [read_manifest(folder / "manifest.json")[0] for folder in folders]
        paths = [folder / "samples.parquet" for folder in folders]
        expected = [receipt["samples_sha256"] for receipt in receipts]
        if [sha256(path) for path in paths] != expected:
            raise ValueError("Las muestras no coinciden con sus huellas confirmadas")
        count += _compare_samples(paths)
        if [sha256(path) for path in paths] != expected:
            raise ValueError("Las muestras han cambiado durante la comparación")
    if count != expected_samples or count != manifests[1][0]["samples"]:
        raise ValueError("El recuento leído no coincide con el manifiesto")
    result = dict(
        schema_version=1,
        assets=len(assets),
        samples=count,
        same_inputs=True,
        different_columns_allowed=["charts", "chart_hash"],
        configuration_sha256=[item[1] for item in configurations],
        manifest_sha256=[item[1] for item in manifests],
        batch_rows=128,
        max_batch_bytes=MAX_BYTES,
    )
    if output is not None:
        safe_destination(output)
        atomic_json(output / "summary.json", result)
    return result


def _stat(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def _track(path, expected, tracked):
    safe_destination(path)
    before = _stat(path)
    if _artifact(path, path.parent) != expected or _stat(path) != before:
        raise ValueError(f"Ha cambiado la huella del origen: {path.name}")
    tracked[path] = expected, before


def _verify(tracked):
    for path, (digest, stat) in tracked.items():
        if _stat(path) != stat or _artifact(path, path.parent) != digest or _stat(path) != stat:
            raise ValueError(f"Un archivo de origen cambió durante la derivación: {path.name}")


def _read(path, tracked):
    value, digest = read_manifest(path)
    _track(path, digest, tracked)
    return value, digest


def _calendars(configuration, clocks):
    contract = check_price_window_contract(configuration.get("price_window"))
    result = {}
    for market, declared in contract["calendars"].items():
        clock = (clocks or {}).get(market) or MarketClock(
            market, declared["start"], declared["end"]
        )
        if clock.market != market or declared != dict(
            start=clock.days[0].isoformat(),
            end=clock.days[-1].isoformat(),
            decisions_sha256=calendar_digest(clock),
        ):
            raise ValueError("La identidad del calendario no coincide con la edición")
        result[market] = clock
    return result


def _prices(path, clock, contract, timing):
    started = time.perf_counter()
    table = read_bounded_table(path, max_rows=MAX_ROWS, max_bytes=MAX_PARTITION_BYTES)
    timing["read"] += time.perf_counter() - started
    started = time.perf_counter()
    table = table.select(["session", "available_at", "open", "high", "low", "close"])
    if any(column.null_count for column in table.columns):
        raise ValueError("Los precios preparados contienen campos ausentes")
    values = table.to_pydict()
    positions = {day.isoformat(): i for i, day in enumerate(clock.days)}
    try:
        indices = np.array([positions[day] for day in values["session"]], dtype=np.int64)
    except KeyError as error:
        raise ValueError("Los precios contienen una sesión ajena al calendario") from error
    absent = absent_positions(clock, contract["market_absent_sessions"][clock.market])
    check_row_positions(indices, absent)
    if any(day >= "2024-01-01" for day in values["session"]) or any(
        aware(available) > clock.decisions[index]
        for available, index in zip(values["available_at"], indices, strict=True)
    ):
        raise ValueError("Los precios contienen futuro o abren la reserva final")
    ohlc = np.column_stack([values[name] for name in ("open", "high", "low", "close")])
    timing["price_preparation"] += time.perf_counter() - started
    return values, ohlc, indices, absent


def _replace_batches(path, prices, clock, ordering_rtol, batch_rows, timing, memory):
    values, ohlc, indices, absent = prices
    previous = -1
    started = time.perf_counter()
    with pq.ParquetFile(path) as file:
        if file.metadata.num_rows > MAX_ROWS or any(
            file.metadata.row_group(i).total_byte_size > MAX_PARTITION_BYTES
            for i in range(file.num_row_groups)
        ):
            raise ValueError("Las muestras superan el presupuesto de lectura")
        if file.schema_arrow.field("charts").type != pa.list_(pa.float32(), WIDTH):
            raise ValueError("La representación original debe tener 512 valores float32")
        stream = file.iter_batches(batch_size=batch_rows, use_threads=False)
        timing["read"] += time.perf_counter() - started
        while True:
            started = time.perf_counter()
            batch = next(stream, None)
            timing["read"] += time.perf_counter() - started
            if batch is None:
                break
            if batch.nbytes > MAX_BYTES:
                raise ValueError("El lote decodificado supera el presupuesto de memoria")
            memory["largest_input_batch_bytes"] = max(
                memory["largest_input_batch_bytes"], batch.nbytes
            )
            started = time.perf_counter()
            hashes, vectors = [], []
            rows = batch.select(
                ["price_end_index", "session", "prediction_at", "input_availability"]
            ).to_pylist()
            for row in rows:
                end = row["price_end_index"]
                if type(end) is not int or not previous < end < len(indices):
                    raise ValueError("El índice de precios es inválido o rompe el orden")
                cutoff = clock.decisions[indices[end]]
                if (
                    row["session"] != values["session"][end]
                    or row["prediction_at"] != cutoff
                    or not isinstance(row["input_availability"], dict)
                    or row["input_availability"]["prices"] != values["available_at"][end]
                    or row["input_availability"]["charts"] != cutoff
                ):
                    raise ValueError("La muestra y su índice no corresponden a la misma sesión")
                present = present_slots(indices, end, 64, absent)
                if present is None:
                    raise ValueError("La ventana contiene una ausencia ajena al contrato")
                digest, vector = geometry_chart(
                    ohlc, end_index=end, present=present, ordering_rtol=ordering_rtol
                )
                hashes.append(digest)
                vectors.append(vector)
                previous = end
            flattened = pa.array(np.asarray(vectors, dtype=np.float32).reshape(-1))
            table = pa.Table.from_batches([batch])
            for name, column in (
                ("charts", list_column(flattened, WIDTH, table.schema.field("charts").type)),
                ("chart_hash", pa.array(hashes, type=table.schema.field("chart_hash").type)),
            ):
                table = table.set_column(
                    table.schema.get_field_index(name), table.schema.field(name), column
                )
            timing["geometry"] += time.perf_counter() - started
            yield table
        if not file.metadata.num_rows:
            yield pa.Table.from_batches([], schema=file.schema_arrow)


def _asset_source(folder, raw, item, configuration, local):
    market, symbol = item["market"], item["symbol"]
    old_config, config_hash = _read(folder / "configuration.json", local)
    receipt, receipt_hash = _read(folder / "manifest.json", local)
    origin, origin_hash = _read(raw / "manifest.json", local)
    if (
        receipt.get("fingerprint") != _digest(old_config)
        or item.get("fingerprint") != receipt["fingerprint"]
        or item["samples"] != receipt.get("samples")
        or old_config.get("prepared_manifest_sha256") != origin_hash
        or receipt.get("prepared_fingerprint") != origin.get("fingerprint")
        or origin.get("policy", {}).get("cutoff", "9999") > "2023-12-31"
        or origin.get("policy", {}).get("calendar") != old_config.get("calendar")
        or old_config.get("calendar")
        != configuration["price_window"]["calendars"][market]["decisions_sha256"]
        or old_config.get("encoders_sha256") != _digest(configuration["encoders"])
        or receipt.get("encoders") != configuration["encoders"]
        or any(
            record.get("market") != market or record.get("symbol") != symbol
            for record in (old_config, receipt, origin)
        )
        or any(
            record.get("price_window") != configuration["price_window"]
            for record in (old_config, receipt)
        )
    ):
        raise ValueError("El activo no conserva su preparación, calendario o representación")
    sample_path, price_path = folder / "samples.parquet", raw / "prices.parquet"
    _track(sample_path, receipt["samples_sha256"], local)
    _track(price_path, origin["artifacts"]["prices.parquet"], local)
    factor_hash = receipt.get("company_factors_sha256")
    if factor_hash is not None:
        _track(folder / "company-factors.parquet", factor_hash, local)
    lineage = dict(
        source_configuration_sha256=config_hash,
        source_manifest_sha256=receipt_hash,
        source_samples_sha256=receipt["samples_sha256"],
        prepared_manifest_sha256=origin_hash,
        prices_sha256=origin["artifacts"]["prices.parquet"],
    )
    return old_config, receipt, origin, lineage


def _asset(
    source, prepared, stage, item, configuration, representation, clock, batch_rows, tracked
):
    market, symbol = item["market"], item["symbol"]
    folder, raw, target = [
        root / market / symbol for root in (source / "samples", prepared, stage / "samples")
    ]
    for path in (folder, raw, target):
        safe_destination(path)
    local = {}
    old_config, receipt, origin, lineage = _asset_source(folder, raw, item, configuration, local)
    sample_path, price_path = folder / "samples.parquet", raw / "prices.parquet"
    factor_hash = receipt.get("company_factors_sha256")
    new_config = deepcopy(old_config)
    new_config["encoders_sha256"] = _digest(representation["encoders"])
    new_config["code"].update(representation["code"])
    lineage["code"] = representation["code"]
    new_config["chart_derivation"] = lineage
    if (target / "manifest.json").exists():
        result, _ = read_manifest(target / "manifest.json")
        configured, _ = read_manifest(target / "configuration.json")
        if (
            configured != new_config
            or result.get("fingerprint") != _digest(new_config)
            or result.get("chart_derivation", {}).get("source") != lineage
            or result.get("encoders") != representation["encoders"]
            or result.get("representation_code") != new_config["code"]
            or result.get("samples") != receipt["samples"]
        ):
            raise ValueError("La recuperación pertenece a otro origen o representación")
        if (
            _artifact(target / "samples.parquet", target) != result["samples_sha256"]
            or factor_hash is not None
            and _artifact(target / "company-factors.parquet", target) != factor_hash
        ):
            raise ValueError("La recuperación contiene un archivo cambiado")
        _verify(local)
        tracked.update(local)
        return result, True
    timing = dict(read=0.0, price_preparation=0.0, geometry=0.0, write=0.0, factor_copy=0.0)
    memory = dict(largest_input_batch_bytes=0)
    prices = _prices(price_path, clock, configuration["price_window"], timing)
    began = time.perf_counter()
    before = timing.copy()
    count = atomic_parquet_batches(
        target / "samples.parquet",
        _replace_batches(
            sample_path,
            prices,
            clock,
            origin.get("price_audit", {}).get("ordering_rtol", 0.0),
            batch_rows,
            timing,
            memory,
        ),
    )
    timing["write"] = max(
        0.0, time.perf_counter() - began - (timing["read"] - before["read"]) - timing["geometry"]
    )
    if count != receipt["samples"]:
        raise ValueError("El número de filas no coincide con el recibo original")
    if factor_hash is not None:
        safe_destination(target / "company-factors.parquet")
        began = time.perf_counter()
        shutil.copyfile(folder / "company-factors.parquet", target / "company-factors.parquet")
        with (target / "company-factors.parquet").open("rb") as stream:
            os.fsync(stream.fileno())
        timing["factor_copy"] = time.perf_counter() - began
        if _artifact(target / "company-factors.parquet", target) != factor_hash:
            raise ValueError("La copia de factores no conserva su huella")
    _verify(local)
    result = dict(
        receipt,
        fingerprint=_digest(new_config),
        encoders=representation["encoders"],
        representation_code=new_config["code"],
        samples_sha256=sha256(target / "samples.parquet"),
        chart_derivation=dict(source=lineage, timing_seconds=timing, **memory),
    )
    atomic_json(target / "configuration.json", new_config)
    atomic_json(target / "manifest.json", result)
    tracked.update(local)
    return result, False


def _code():
    return {
        name: sha256(Path(__file__).with_name(name))
        for name in (
            "candle_geometry_edition.py",
            "candle_geometry_selection.py",
            "candle_geometry.py",
            "price_windows.py",
            "prices.py",
            "batches.py",
            "cohort_files.py",
            "storage.py",
        )
    }


def derive_geometry_edition(
    source,
    output,
    *,
    batch_rows=128,
    clocks=None,
    selection=None,
    reserve_bytes=selection_policy.DEFAULT_RESERVE_BYTES,
    estimate_only=False,
):
    """Copiar todas las filas a otro destino y cambiar solo las dos columnas de velas.

    Reanuda activos confirmados en una carpeta provisional identificada. El destino final
    aparece únicamente tras cotejar las columnas conservadas y volver a verificar el origen.
    No carga modelos, etiquetas ni cachés de texto y rechaza decisiones desde 2024.
    `selection` identifica candidatos de un origen completo sin filtrar sus filas.
    `estimate_only` inspecciona metadatos y devuelve el presupuesto sin crear el destino.
    """
    if type(batch_rows) is not int or not 1 <= batch_rows <= 512:
        raise ValueError("El lote debe contener entre 1 y 512 filas")
    if type(estimate_only) is not bool:
        raise ValueError("La opción de estimación debe ser booleana")
    began = time.perf_counter()
    source, output = Path(source), Path(output)
    safe_destination(source)
    safe_destination(output)
    if output.exists():
        raise FileExistsError("La edición de destino ya existe")
    tracked = {}
    configuration, config_hash = _read(source / "configuration.json", tracked)
    manifest, manifest_hash = _read(source / "manifest.json", tracked)
    cohort_identity(manifest, input_policy=HISTORICAL_MASKED)
    if (
        manifest.get("kind") != "materialized_corpus"
        or manifest.get("cohort_complete") is not True
        or manifest.get("configuration") != configuration
        or configuration.get("context_sessions") != 64
        or len(manifest["coverage"]) > 10_000
        or configuration["encoders"].get("image_model") == MODEL
    ):
        raise ValueError(
            "El origen debe ser una edición histórica completa con otra representación"
        )
    for item in manifest["coverage"]:
        if (
            item["market"] not in {"US", "CN"}
            or not re.fullmatch(r"[A-Z0-9.^_=\-]{1,64}", item["symbol"])
            or item["symbol"] in {".", ".."}
        ):
            raise ValueError("La identidad del activo no es válida")
    prepared = Path(configuration["prepared_root"])
    for root in (source, prepared, Path("dataset").resolve()):
        outside_source(root, output)
        outside_source(output, root)
    calendars = _calendars(configuration, clocks)
    if set(calendars) != set(manifest["markets"]):
        raise ValueError("El calendario no cubre exactamente los mercados de la edición")
    selected = None
    if selection is not None:
        path = Path(selection)
        request, request_hash = read_manifest(path, maximum=1024**2)
        _track(path, request_hash, tracked)
        selected = selection_policy.selection_record(
            request, request_hash, manifest, manifest_hash, config_hash
        )
        manifest = selection_policy.project_manifest(manifest, selected)
    plan = selection_policy.storage_plan(
        source, output, manifest, batch_rows=batch_rows, reserve_bytes=reserve_bytes
    )
    plan.update(
        source_manifest_sha256=manifest_hash,
        source_configuration_sha256=config_hash,
        selection=selected,
        derivation_code=_code(),
        geometry_encoder_spec_sha256=_digest(geometry_encoder_spec(configuration["encoders"])),
    )
    if estimate_only:
        _verify(tracked)
        return plan
    stage = output.with_name(f".{output.name}.geometry-staging")
    safe_destination(stage)
    if not plan["fits_in_free_space"] and not stage.exists():
        raise OSError("El espacio libre no alcanza la estimación y la reserva de disco")
    representation = dict(encoders=geometry_encoder_spec(configuration["encoders"]), code=_code())
    identity = dict(
        version=1,
        source=str(source.resolve()),
        configuration_sha256=config_hash,
        manifest_sha256=manifest_hash,
        representation=MODEL,
        code=representation["code"],
        batch_rows=batch_rows,
    )
    if selected is not None:
        identity["selection"] = selected
    lock_path = output.with_name(f".{output.name}.geometry.lock")
    safe_destination(stage)
    safe_destination(lock_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise FileExistsError("La edición de destino ya existe")
        if stage.exists():
            if read_manifest(stage / "derivation.json")[0] != identity:
                raise ValueError("La recuperación corresponde a otra derivación")
        else:
            stage.mkdir()
            atomic_json(stage / "derivation.json", identity)
        new_config = deepcopy(configuration)
        new_config["encoders"] = representation["encoders"]
        new_config["code"].update(representation["code"])
        remaining = (
            sum(
                row["estimated_output_bytes"]
                for row in plan["assets"]
                if not (
                    stage / "samples" / row["market"] / row["symbol"] / "manifest.json"
                ).exists()
            )
            + plan["root_metadata_bytes"]
        )
        if selection_policy.free_disk_bytes(output.parent) < reserve_bytes + remaining:
            raise OSError("El espacio libre no conserva la reserva de disco")
        atomic_json(stage / "configuration.json", new_config)
        estimated = {
            (row["market"], row["symbol"]): row["estimated_output_bytes"] for row in plan["assets"]
        }
        coverage, timings, reused = (
            [],
            dict(read=0.0, price_preparation=0.0, geometry=0.0, write=0.0, factor_copy=0.0),
            0,
        )
        for item in manifest["coverage"]:
            item = dict(item)
            if item["state"] == "encoded":
                if selection_policy.free_disk_bytes(output.parent) < reserve_bytes + remaining:
                    raise OSError("El espacio libre no conserva la reserva de disco")
                result, was_reused = _asset(
                    source,
                    prepared,
                    stage,
                    item,
                    configuration,
                    representation,
                    calendars[item["market"]],
                    batch_rows,
                    tracked,
                )
                item["fingerprint"] = result["fingerprint"]
                reused += was_reused
                if not was_reused:
                    remaining -= estimated[(item["market"], item["symbol"])]
                    for key in timings:
                        timings[key] += result["chart_derivation"]["timing_seconds"][key]
            coverage.append(item)
        result = dict(
            manifest,
            configuration=new_config,
            coverage=coverage,
            samples_root=str((output / "samples").resolve()),
        )
        atomic_json(stage / "manifest.json", result)
        started = time.perf_counter()
        compared = compare_representations(source, stage)
        timings["comparison"] = time.perf_counter() - started
        atomic_json(stage / "comparison-manifest.json", result)
        started = time.perf_counter()
        _verify(tracked)
        timings["final_source_verification"] = time.perf_counter() - started
        result["chart_derivation"] = dict(
            source=identity,
            storage_plan=plan,
            comparison=compared,
            reused_assets=reused,
            timing_seconds=timings,
            before_publication_seconds=time.perf_counter() - began,
            peak_process_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            unique_input_file_bytes=sum(stat[2] for _, stat in tracked.values()),
            samples_output_bytes=sum(
                (stage / "samples" / a["market"] / a["symbol"] / "samples.parquet").stat().st_size
                for a in manifest["coverage"]
                if a["state"] == "encoded"
            ),
            limits=dict(
                batch_rows=batch_rows,
                batch_bytes=MAX_BYTES,
                partition_bytes=MAX_PARTITION_BYTES,
                rows_per_asset=MAX_ROWS,
                assets=10_000,
            ),
        )
        atomic_json(stage / "manifest.json", result)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=128)
    parser.add_argument("--selection", type=Path, help="JSON con la lista explícita de activos")
    parser.add_argument("--reserve-gib", type=int, default=16)
    parser.add_argument("--estimate-only", action="store_true")
    args = parser.parse_args(argv)
    result = derive_geometry_edition(
        args.source,
        args.output,
        batch_rows=args.batch_rows,
        selection=args.selection,
        reserve_bytes=args.reserve_gib * 1024**3,
        estimate_only=args.estimate_only,
    )
    print(
        json.dumps(
            result
            if args.estimate_only
            else dict(samples=result["samples"], **result["chart_derivation"]),
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
