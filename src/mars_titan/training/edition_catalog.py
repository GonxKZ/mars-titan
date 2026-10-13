"""Catálogo Parquet de la edición histórica y de sus objetivos, con una fila por muestra.

El catálogo dice qué hay en cada muestra sin copiar sus vectores. Cada fila se identifica
con (`market`, `asset_id`, `prediction_at`), la misma clave que usan las predicciones y
`integrity.row_identity`, y lleva:

- la sesión del mercado y el instante de la predicción,
- la disponibilidad de cada bloque y la de todas las entradas, con la fila de precios que
  cierra la ventana, igual que la calcula la lectura de las vistas,
- los bits de presencia de precios, noticias, gráficos, fundamentales y macro,
- el motivo, la partición y la madurez del objetivo, sin su valor,
- el archivo, el grupo de filas y la posición de origen de la muestra y de su etiqueta.

Ninguna columna de entrada es posterior al instante de la predicción: la construcción se
detiene si una disponibilidad lo es. La madurez del objetivo sí es posterior por definición.
Describe cuándo se conoce la etiqueta y nunca su valor, así que elegir filas por ella no
usa el resultado. El catálogo no contiene filas de 2024 ni las lee. Antes de leer un archivo
de muestras se comprueba con las estadísticas de cada grupo que termina antes del test
reservado.

La construcción recorre los activos en el orden del manifiesto de objetivos, lee de cada
archivo solo las columnas de metadatos y escribe grupos de filas de tamaño fijo, así que la
memoria no crece con la edición. `verify_catalog` vuelve a leer las fuentes con Arrow, sin
pasar por el constructor, y compara recuentos, huellas y filas.
"""

import argparse
import json
import resource
import tempfile
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.input_policy import HISTORICAL_MASKED, MODALITIES, policy_identity
from mars_titan.data.macro_coverage import _publish_directory
from mars_titan.data.storage import atomic_json, outside_source, sha256
from mars_titan.data.temporal import MarketClock

from .cohort_contract import representation_hash, validate_cohort_rows
from .corpus_inputs import (
    CorpusDataset,
    _availability,
    _historical_times,
    _times,
    presence_bits,
)
from .temporal_corpus import _annual_candidates, annual_boundary

SCHEMA_VERSION = 1
KIND = "edition_catalog"
CATALOG_FILE = "catalog.parquet"
FINAL_TEST_START = "2024-01-01"
# Grupos de un millón de filas: unos 60 MB decodificados y lecturas por columna eficaces.
ROW_GROUP_ROWS = 1 << 20
STAMP = pa.timestamp("us", tz="UTC")
SCHEMA = pa.schema(
    [
        ("market", pa.string()),
        ("asset_id", pa.string()),
        ("session", pa.date32()),
        ("prediction_at", STAMP),
        *[(f"{name}_available_at", STAMP) for name in MODALITIES],
        ("inputs_available_at", STAMP),
        *[(f"has_{name}", pa.bool_()) for name in MODALITIES],
        ("news_count", pa.int32()),
        ("label_reason", pa.string()),
        ("annual_boundary_label", pa.bool_()),
        ("target_partition", pa.string()),
        ("target_available_at", STAMP),
        ("samples_file", pa.string()),
        ("sample_row", pa.int32()),
        ("row_group", pa.int32()),
        ("row_in_group", pa.int32()),
        ("label_row", pa.int32()),
    ]
)
KEYS = ("market", "asset_id", "prediction_at")
# Columnas de metadatos de las muestras. Los vectores no se leen nunca.
SAMPLE_COLUMNS = (
    "cohort_id",
    "prediction_at",
    "price_end_index",
    "input_availability",
    "presence",
    "news_count",
)
LABEL_COLUMNS = ("sample_row", "prediction_at", "target_available_at", "target", "reason")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _micros(day):
    return int(np.datetime64(day, "us").astype(np.int64))


def _before_test(file, path):
    """Comprobar con las estadísticas de cada grupo que no hay filas de 2024 antes de leer."""
    index = file.schema_arrow.get_field_index("prediction_at")
    _require(index >= 0, f"{path} no tiene instante de predicción")
    logical = file.schema.column(index).logical_type.to_json()
    _require(
        '"isAdjustedToUTC": true' in logical and '"timeUnit": "microseconds"' in logical,
        f"{path} no guarda la predicción en microsegundos UTC",
    )
    limit = _micros(FINAL_TEST_START)
    for group in range(file.num_row_groups):
        metadata = file.metadata.row_group(group)
        if not metadata.num_rows:
            continue
        statistics = metadata.column(index).statistics
        _require(
            statistics is not None and statistics.has_min_max,
            f"{path} no permite comprobar el test reservado sin leer sus filas",
        )
        _require(statistics.max_raw < limit, f"{path} contiene filas del test reservado")


def _session_days(market):
    """Día de mercado de cada decisión del calendario, como date32."""
    clock = MarketClock(market, "1999-01-01", "2024-01-05")
    moments = np.array([round(t.timestamp() * 1_000_000) for t in clock.decisions], np.int64)
    days = np.array([day.toordinal() - 719163 for day in clock.days], dtype=np.int32)
    return moments, days


def _sessions(calendar, prediction):
    moments, days = calendar
    positions = np.searchsorted(moments, prediction)
    _require(
        np.all(positions < len(moments)) and np.array_equal(moments[positions], prediction),
        "Una predicción no es una decisión del calendario del mercado",
    )
    return days[positions]


def _stamp(values):
    return pa.array(values, type=pa.int64()).cast(STAMP)


def _sample_table(dataset, asset, calendar):
    """Metadatos de todas las muestras de un activo, con su posición en el archivo."""
    prices, price_available = dataset._prices(asset)
    path = dataset._file(asset, "samples")
    with pq.ParquetFile(path) as file:
        _before_test(file, path)
        names = set(file.schema_arrow.names)
        _require(set(SAMPLE_COLUMNS) <= names, f"{path} no tiene los metadatos de la edición")
        columns = [*SAMPLE_COLUMNS, *(["session"] if "session" in names else [])]
        sizes = [file.metadata.row_group(g).num_rows for g in range(file.num_row_groups)]
        table = file.read(columns=columns, use_threads=False)
    _require(table.num_rows == sum(sizes) and table.num_rows <= 1_000_000, f"{path} no cuadra")
    validate_cohort_rows(table, dataset.cohort)
    prediction = _historical_times(table)
    presence = presence_bits(table)
    available, valid = _availability(table, presence=presence)
    ends = table["price_end_index"]
    _require(pa.types.is_integer(ends.type) and not ends.null_count, "Índice de precios inválido")
    ends = ends.to_numpy()
    _require(((ends >= 0) & (ends < len(prices))).all(), "La ventana no tiene sus precios")
    available = np.maximum(available, price_available[ends])
    _require(
        valid.all() and not (available > prediction).any(),
        f"{path} tiene una entrada con disponibilidad ausente o futura",
    )
    sessions = _sessions(calendar, prediction)
    if "session" in table.column_names:
        declared = pc.cast(table["session"], pa.date32()).to_numpy(zero_copy_only=False)
        _require(
            np.array_equal(declared.astype("datetime64[D]").astype(np.int32), sessions),
            f"{path} declara una sesión distinta de la de su decisión",
        )
    group = np.repeat(np.arange(len(sizes), dtype=np.int32), sizes)
    within = np.concatenate([np.arange(size, dtype=np.int32) for size in sizes] or [[]])
    struct = table["input_availability"].combine_chunks()
    columns = dict(
        market=pa.array([asset["market"]] * table.num_rows, pa.string()),
        asset_id=pa.array([asset["symbol"]] * table.num_rows, pa.string()),
        session=pa.array(sessions, pa.int32()).cast(pa.date32()),
        prediction_at=_stamp(prediction),
    )
    for name in MODALITIES:
        columns[f"{name}_available_at"] = struct.field(name).cast(STAMP)
    columns["inputs_available_at"] = _stamp(available)
    for index, name in enumerate(MODALITIES):
        columns[f"has_{name}"] = pa.array(presence[:, index], pa.bool_())
    columns["news_count"] = table["news_count"].combine_chunks().cast(pa.int32())
    positions = dict(
        samples_file=pa.array(
            [f"{asset['market']}/{asset['symbol']}/samples.parquet"] * table.num_rows, pa.string()
        ),
        sample_row=pa.array(np.arange(table.num_rows, dtype=np.int32)),
        row_group=pa.array(group),
        row_in_group=pa.array(within, pa.int32()),
    )
    return prediction, columns, positions


def _label_columns(dataset, asset, prediction, boundary):
    """Motivo, partición y madurez de cada muestra, sin el valor del objetivo."""
    path = dataset._file(asset, "labels")
    with pq.ParquetFile(path) as file:
        _before_test(file, path)
        _require(set(LABEL_COLUMNS) | {"partition"} <= set(file.schema_arrow.names), "Etiquetas")
        _require(file.metadata.num_rows <= 1_000_000, f"{path} supera su presupuesto")
        labels = file.read(columns=[*LABEL_COLUMNS, "partition"], use_threads=False)
    validate_rows = labels["sample_row"]
    _require(not validate_rows.null_count, "Una etiqueta no identifica su muestra")
    rows = validate_rows.to_numpy()
    _require(
        len(np.unique(rows)) == len(rows) and ((rows >= 0) & (rows < len(prediction))).all(),
        f"{path} repite o pierde muestras",
    )
    label_prediction = _times(labels["prediction_at"])
    _require(np.array_equal(prediction[rows], label_prediction), f"{path} no está alineado")
    reasons = np.asarray(labels["reason"].to_pylist(), dtype="U40")
    maturity = _times(labels["target_available_at"].fill_null(pa.scalar(0, STAMP)))
    recovered = _annual_candidates(labels, label_prediction, maturity, reasons, boundary)
    # Posición de la etiqueta de cada muestra. Una muestra sin etiqueta queda nula.
    index = np.full(len(prediction), -1, dtype=np.int64)
    index[rows] = np.arange(len(rows))
    take = pa.array(index, mask=index < 0)
    return dict(
        label_reason=labels["reason"].combine_chunks().take(take),
        annual_boundary_label=pa.array(recovered).take(take),
        target_partition=labels["partition"].combine_chunks().take(take),
        target_available_at=labels["target_available_at"].combine_chunks().cast(STAMP).take(take),
        label_row=pa.array(index.astype(np.int32), mask=index < 0),
    )


def _asset_table(dataset, asset, calendars, boundaries):
    market = asset["market"]
    prediction, samples, positions = _sample_table(dataset, asset, calendars[market])
    labels = _label_columns(dataset, asset, prediction, boundaries[market])
    columns = {**samples, **labels, **positions}
    return pa.Table.from_arrays([columns[name] for name in SCHEMA.names], schema=SCHEMA)


def _check_sources(dataset, edition, edition_sha256):
    """La edición y los objetivos deben ser una pareja confirmada y completa."""
    meta = dataset.manifest
    _require(dataset.temporal is None, "El catálogo parte de los objetivos completos")
    _require(
        meta.get("configuration", {}).get("source_manifest_sha256") == edition_sha256,
        "Los objetivos no proceden de esta edición",
    )
    _require(
        edition.get("kind") == "materialized_corpus"
        and edition.get("final_test_opened") is False
        and edition.get("input_policy") == HISTORICAL_MASKED
        and edition.get("cohort_complete") is True
        and edition.get("failed_assets") == 0
        and Path(edition.get("samples_root", "")).resolve() == dataset.roots["samples"]
        and edition.get("samples") == meta.get("samples"),
        "La edición no está completa o no comparte raíz y recuento con los objetivos",
    )
    _require(
        sum(asset["samples"] for asset in dataset.assets) == meta["samples"],
        "Los activos de los objetivos no suman las muestras declaradas",
    )


def _summary(table, counts):
    """Acumular los recuentos de un activo en el resumen del catálogo."""
    if not table.num_rows:
        return
    market = table["market"][0].as_py()
    counts["markets"][market] += table.num_rows
    years = pc.year(table["session"]).to_numpy(zero_copy_only=False)
    for year, total in zip(*np.unique(years, return_counts=True), strict=True):
        counts["years"][f"{market}:{year}"] += int(total)
    reasons = table["label_reason"].fill_null("without_label").to_pylist()
    counts["label_reasons"].update(reasons)
    counts["target_partitions"].update(p for p in table["target_partition"].to_pylist() if p)
    counts["annual_boundary_labels"]["rows"] += int(
        pc.sum(table["annual_boundary_label"]).as_py() or 0
    )
    for name in MODALITIES:
        counts["presence"][name] += int(pc.sum(table[f"has_{name}"]).as_py() or 0)


def _counts_document(counts):
    years = {}
    for key, total in sorted(counts["years"].items()):
        market, year = key.split(":")
        years.setdefault(market, {})[year] = total
    return dict(
        rows=sum(counts["markets"].values()),
        markets=dict(sorted(counts["markets"].items())),
        years=years,
        label_reasons=dict(sorted(counts["label_reasons"].items())),
        target_partitions=dict(sorted(counts["target_partitions"].items())),
        annual_boundary_labels=counts["annual_boundary_labels"]["rows"],
        presence={name: counts["presence"][name] for name in MODALITIES},
    )


def build_catalog(targets, edition, output, *, row_group_rows=ROW_GROUP_ROWS):
    """Escribir el catálogo y su manifiesto en una carpeta nueva publicada de una vez."""
    began = time.perf_counter()
    targets, edition, output = Path(targets), Path(edition), Path(output)
    _require(type(row_group_rows) is int and row_group_rows >= 1, "Tamaño de grupo inválido")
    if output.exists() or output.is_symlink():
        raise FileExistsError("El catálogo ya existe")
    safe_destination(output)
    dataset = CorpusDataset(targets, input_policy=HISTORICAL_MASKED)
    edition_meta, edition_sha256 = read_manifest(edition)
    _check_sources(dataset, edition_meta, edition_sha256)
    for source in (*dataset.roots.values(), edition.parent.resolve(), Path("dataset")):
        outside_source(source, output)
    markets = sorted({asset["market"] for asset in dataset.assets})
    calendars = {market: _session_days(market) for market in markets}
    boundaries = {market: annual_boundary(market) for market in markets}
    counts = dict(
        markets=Counter(),
        years=Counter(),
        label_reasons=Counter(),
        target_partitions=Counter(),
        annual_boundary_labels=Counter(),
        presence=Counter(),
    )
    assets, peak_buffer = [], 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "catalog"
        stage.mkdir()
        path = stage / CATALOG_FILE
        pending, buffered = [], 0
        with pq.ParquetWriter(path, SCHEMA, compression="zstd") as writer:
            for asset in dataset.assets:
                table = _asset_table(dataset, asset, calendars, boundaries)
                _require(table.num_rows == asset["samples"], "Un activo no conserva sus muestras")
                _summary(table, counts)
                assets.append(
                    dict(
                        market=asset["market"],
                        asset_id=asset["symbol"],
                        rows=table.num_rows,
                        **{
                            f"{kind}_sha256": asset[f"{kind}_sha256"]
                            for kind in ("samples", "labels", "prices")
                        },
                    )
                )
                pending.append(table)
                buffered += table.num_rows
                peak_buffer = max(peak_buffer, sum(part.nbytes for part in pending))
                if buffered >= row_group_rows:
                    joined = pa.concat_tables(pending).combine_chunks()
                    full = buffered - buffered % row_group_rows
                    writer.write_table(joined.slice(0, full), row_group_size=row_group_rows)
                    pending = [joined.slice(full)] if full < buffered else []
                    buffered -= full
            if buffered:
                writer.write_table(pa.concat_tables(pending), row_group_size=row_group_rows)
        # Un archivo que cambió durante la lectura ya no corresponde a su huella confirmada.
        for asset in dataset.assets:
            for kind in ("prices", "samples", "labels"):
                dataset._file(asset, kind)
        _require(sha256(edition) == edition_sha256, "La edición cambió durante el catálogo")
        _require(sha256(targets) == dataset.identity, "Los objetivos cambiaron durante el catálogo")
        summary = _counts_document(counts)
        _require(summary["rows"] == dataset.manifest["samples"], "El catálogo pierde muestras")
        _require(
            summary["target_partitions"] == dataset.manifest["counts"],
            "Las particiones del catálogo no coinciden con los objetivos",
        )
        metadata = pq.ParquetFile(path).metadata
        manifest = dict(
            schema_version=SCHEMA_VERSION,
            kind=KIND,
            **policy_identity(HISTORICAL_MASKED),
            keys=list(KEYS),
            columns=SCHEMA.names,
            catalog=dict(
                file=CATALOG_FILE,
                sha256=sha256(path),
                bytes=path.stat().st_size,
                rows=metadata.num_rows,
                row_groups=metadata.num_row_groups,
                row_group_rows=row_group_rows,
            ),
            edition=dict(path=str(edition.resolve()), sha256=edition_sha256),
            targets=dict(path=str(targets.resolve()), sha256=dataset.identity),
            roots={name: str(root) for name, root in sorted(dataset.roots.items())},
            representation_sha256=representation_hash(
                dataset.manifest["representation"], input_policy=HISTORICAL_MASKED
            ),
            final_test=dict(start=FINAL_TEST_START, opened=False, rows=0),
            target_value_included=False,
            counts=summary,
            assets=assets,
            build=dict(
                elapsed_seconds=time.perf_counter() - began,
                peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                peak_buffer_bytes=peak_buffer,
                pyarrow=pa.__version__,
                numpy=np.__version__,
                built_at=datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
        atomic_json(stage / "manifest.json", manifest)
        _publish_directory(stage, output)
    return manifest


def read_catalog_manifest(path):
    """Manifiesto del catálogo con su contrato y la huella del archivo de filas comprobados."""
    path = Path(path)
    manifest, digest = read_manifest(path)
    _require(
        manifest.get("schema_version") == SCHEMA_VERSION
        and manifest.get("kind") == KIND
        and manifest.get("columns") == SCHEMA.names
        and manifest.get("keys") == list(KEYS)
        and manifest.get("target_value_included") is False
        and manifest.get("final_test") == dict(start=FINAL_TEST_START, opened=False, rows=0),
        "El manifiesto del catálogo no cumple su contrato",
    )
    catalog = path.parent / manifest["catalog"]["file"]
    _require(
        catalog.is_file()
        and not catalog.is_symlink()
        and catalog.parent == path.parent
        and sha256(catalog) == manifest["catalog"]["sha256"],
        "El archivo del catálogo no corresponde a su manifiesto",
    )
    return manifest, digest, catalog


def asset_slices(file, sizes, columns):
    """Filas consecutivas del catálogo de cada activo, leídas en flujo."""
    pending, buffered = [], 0
    batches = file.iter_batches(batch_size=1 << 16, columns=columns, use_threads=False)
    for size in sizes:
        while buffered < size:
            batch = next(batches, None)
            _require(batch is not None, "El catálogo termina antes que sus activos")
            pending.append(batch)
            buffered += batch.num_rows
        table = pa.Table.from_batches(pending).combine_chunks()
        yield table.slice(0, size)
        rest = table.slice(size)
        pending, buffered = ([rest.to_batches()[0]] if rest.num_rows else []), rest.num_rows
    _require(next(batches, None) is None and not buffered, "El catálogo tiene filas de más")


def _source(root, asset, kind, expected):
    path = Path(root) / asset["market"] / asset["symbol"] / f"{kind}.parquet"
    _require(
        path.is_file() and not path.is_symlink() and sha256(path) == expected,
        f"{path} no conserva la huella confirmada",
    )
    return path


def _array(values):
    return values.combine_chunks() if isinstance(values, pa.ChunkedArray) else values


def _equal(left, right):
    """Misma secuencia de valores, con la columna del catálogo en el tipo de la fuente."""
    left, right = _array(left), _array(right)
    return (left if left.type == right.type else left.cast(right.type)).equals(right)


def verify_catalog(path):
    """Contrastar el catálogo con la edición y los objetivos por un camino independiente.

    Vuelve a calcular la huella de cada archivo de muestras y de etiquetas, lee con Arrow sus
    columnas de clave, presencia y objetivo, y las compara fila a fila con el catálogo. No
    usa el constructor ni la lectura de entrenamiento.
    """
    began = time.perf_counter()
    path = Path(path)
    manifest, digest, catalog = read_catalog_manifest(path)
    targets, targets_sha256 = read_manifest(Path(manifest["targets"]["path"]))
    edition, edition_sha256 = read_manifest(Path(manifest["edition"]["path"]))
    _require(
        targets_sha256 == manifest["targets"]["sha256"]
        and edition_sha256 == manifest["edition"]["sha256"]
        and targets["configuration"]["source_manifest_sha256"] == edition_sha256,
        "La edición o los objetivos ya no son los del catálogo",
    )
    _require(
        targets["samples"] == edition["samples"] == manifest["catalog"]["rows"]
        and targets["final_test_opened"] is False
        and edition["final_test_opened"] is False,
        "Los recuentos de la edición, los objetivos y el catálogo no coinciden",
    )
    declared = [(a["market"], a["asset_id"], a["rows"]) for a in manifest["assets"]]
    _require(
        declared == [(a["market"], a["symbol"], a["samples"]) for a in targets["assets"]],
        "El catálogo no recorre los activos de los objetivos en su orden",
    )
    roots = targets["roots"]
    columns = [
        *KEYS,
        "session",
        *[f"{name}_available_at" for name in MODALITIES],
        "inputs_available_at",
        *[f"has_{name}" for name in MODALITIES],
        "label_reason",
        "target_partition",
        "target_available_at",
        "sample_row",
        "label_row",
    ]
    limit = _micros(FINAL_TEST_START)
    partitions, rows = Counter(), 0
    with pq.ParquetFile(catalog) as file:
        _before_test(file, catalog)
        _require(file.schema_arrow == SCHEMA, "El esquema del catálogo cambió")
        sizes = [a["rows"] for a in manifest["assets"]]
        for asset, table in zip(targets["assets"], asset_slices(file, sizes, columns), strict=True):
            samples_path = _source(roots["samples"], asset, "samples", asset["samples_sha256"])
            labels_path = _source(roots["labels"], asset, "labels", asset["labels_sha256"])
            with pq.ParquetFile(samples_path) as samples_file:
                _before_test(samples_file, samples_path)
                samples = samples_file.read(columns=["prediction_at", "presence"])
            with pq.ParquetFile(labels_path) as labels_file:
                _before_test(labels_file, labels_path)
                labels = labels_file.read(
                    columns=[
                        "sample_row",
                        "prediction_at",
                        "reason",
                        "partition",
                        "target_available_at",
                    ]
                )
            name = f"{asset['market']}/{asset['symbol']}"
            _require(table.num_rows == samples.num_rows, f"{name} no conserva sus muestras")
            _require(
                pc.all(pc.equal(table["market"], asset["market"])).as_py()
                and pc.all(pc.equal(table["asset_id"], asset["symbol"])).as_py()
                and _equal(table["prediction_at"], samples["prediction_at"].combine_chunks()),
                f"{name} no conserva sus claves",
            )
            moments = table["prediction_at"].cast(pa.int64()).to_numpy()
            _require(not len(moments) or moments.max() < limit, f"{name} tiene filas de 2024")
            presence = pc.list_flatten(samples["presence"].combine_chunks()).to_numpy(
                zero_copy_only=False
            )
            presence = presence.reshape(samples.num_rows, len(MODALITIES))
            for index, modality in enumerate(MODALITIES):
                _require(
                    np.array_equal(
                        table[f"has_{modality}"].to_numpy(zero_copy_only=False), presence[:, index]
                    ),
                    f"{name} no conserva la presencia de {modality}",
                )
                available = table[f"{modality}_available_at"].cast(pa.int64())
                _require(
                    pc.all(
                        pc.less_equal(available, table["prediction_at"].cast(pa.int64()))
                    ).as_py()
                    in (True, None),
                    f"{name} tiene una entrada posterior a la predicción",
                )
            _require(
                pc.all(
                    pc.less_equal(
                        table["inputs_available_at"].cast(pa.int64()),
                        table["prediction_at"].cast(pa.int64()),
                    )
                ).as_py()
                in (True, None),
                f"{name} tiene una entrada posterior a la predicción",
            )
            order = labels["sample_row"].to_numpy()
            labelled = table["label_row"].is_valid().to_numpy(zero_copy_only=False)
            take = np.argsort(order)
            _require(
                np.array_equal(np.flatnonzero(labelled), order[take])
                and np.array_equal(table["label_row"].drop_null().to_numpy(), take),
                f"{name} no enlaza sus etiquetas",
            )
            take = pa.array(take)
            for column, source in (
                ("label_reason", "reason"),
                ("target_partition", "partition"),
                ("target_available_at", "target_available_at"),
            ):
                expected = labels[source].combine_chunks().take(take)
                actual = table[column].filter(pa.array(labelled))
                _require(_equal(actual.combine_chunks(), expected), f"{name} cambia {column}")
            _require(
                _equal(
                    table["prediction_at"].filter(pa.array(labelled)).combine_chunks(),
                    labels["prediction_at"].combine_chunks().take(take),
                ),
                f"{name} no alinea etiquetas y muestras",
            )
            partitions.update(p for p in table["target_partition"].to_pylist() if p)
            rows += table.num_rows
    _require(rows == targets["samples"], "El catálogo no tiene todas las muestras")
    _require(
        dict(partitions) == targets["counts"], "Las particiones no coinciden con los objetivos"
    )
    return dict(
        status="verified",
        catalog_manifest_sha256=digest,
        catalog_sha256=manifest["catalog"]["sha256"],
        rows=rows,
        assets=len(targets["assets"]),
        target_partitions=dict(sorted(partitions.items())),
        elapsed_seconds=time.perf_counter() - began,
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Construir el catálogo en una carpeta nueva")
    build.add_argument("--targets", type=Path, required=True, help="Manifiesto de los objetivos")
    build.add_argument("--edition", type=Path, required=True, help="Manifiesto de la edición")
    build.add_argument("--output", type=Path, required=True, help="Carpeta nueva del catálogo")
    verify = commands.add_parser("verify", help="Contrastar el catálogo con sus fuentes")
    verify.add_argument("--catalog", type=Path, required=True, help="Manifiesto del catálogo")
    verify.add_argument("--report", type=Path, required=True, help="Informe JSON nuevo")
    args = parser.parse_args(argv)
    if args.command == "build":
        manifest = build_catalog(args.targets, args.edition, args.output)
        print(json.dumps(dict(rows=manifest["catalog"]["rows"], **manifest["build"])))
        return 0
    if args.report.exists():
        raise FileExistsError("El informe ya existe")
    report = verify_catalog(args.catalog)
    atomic_json(args.report, report)
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
