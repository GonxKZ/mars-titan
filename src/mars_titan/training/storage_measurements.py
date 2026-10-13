"""Medir con tablas sintéticas cuánto ocupan las tablas por fila de la campaña.

Las tablas reproducen el esquema, el orden de filas y los grupos Parquet de cada escritor
con el número real de filas de cada activo, leído del manifiesto de una vista. Las claves
son sintéticas y los valores son ruido gaussiano, así que no se leen objetivos, muestras
ni modelos. El ruido gaussiano no se comprime, y por eso la medida de los decimales es
conservadora frente a predicciones reales.

Se comparan tres disposiciones:

- ``current``: lo que escribe hoy ``data.batches.atomic_parquet_batches`` con el lote de
  cada escritor, un grupo Parquet por lote, zstd del códec por defecto y diccionario en
  todas las columnas.
- ``large_groups``: las mismas columnas, tipos, valores y orden en grupos de hasta
  ``LARGE_GROUP_ROWS`` filas, diccionario solo en texto e instantes, ``byte_stream_split``
  en los decimales y zstd de nivel ``ZSTD_LEVEL``, con las opciones de
  ``data.prediction_files``, que es el formato de la retención v2.
- ``shared_rows``: una tabla de filas por ámbito, ventana y tramo con identificador,
  activo, mercado, instante y objetivo, escrita una sola vez, y por ajuste solo sus
  decimales con la disposición anterior, en el orden común de la tabla de filas. La
  predicción puntual de la cabeza de cuantiles repite los bits de la mediana y el control
  nulo vale cero, así que ninguno de los dos se guarda. Los consumidores de la campaña
  ordenan las filas al leerlas, de modo que el orden común no cambia lo que calculan.
- ``shared_rows_ordered``: lo mismo con el índice de fila en la tabla común cuando el
  orden de escritura no es el común, como en los escritores cronológicos.

Cada medida vuelve a leer lo escrito y exige la misma tabla que la original, con tipos y
bits iguales. Así una disposición solo se mide si es reversible sin pérdida.

También se miden, con las mismas claves, la tabla de predicciones de la etapa de
adaptadores, los agregados por sesión y estrato que sustituirían a las filas de un ajuste
y el índice de observaciones de la familia Titans y de la GRU candidata, escrito con las
mismas opciones de DuckDB que ``environments.corpus_source._ordered_parquet``.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest
from mars_titan.data.prediction_files import (
    KEY_COLUMNS,
    MEDIAN,
    same_table,
    write_large,
)
from mars_titan.data.prediction_files import canonical_order as _canonical_order
from mars_titan.data.prediction_files import shared_rows as canonical_rows

HELD_OUT = ("validation", "calibration", "evaluation")
QUANTILE_COLUMNS = (
    "quantile_0025",
    "quantile_0100",
    "quantile_0500",
    "quantile_0900",
    "quantile_0975",
)
LAYOUTS = ("current", "large_groups", "shared_rows", "shared_rows_ordered")
# Hora de cierre en UTC de cada mercado para separar sus instantes en una vista conjunta.
_CLOSE_HOURS = {"US": 20, "CN": 7}
_DAY = 86_400_000_000


@dataclass(frozen=True)
class Writer:
    """Esquema por fila de un escritor de predicciones, tal como lo fija su código."""

    prediction: pa.DataType
    quantiles: pa.DataType | None
    chronological: bool
    row_group: int
    # Valores calculados en float32, aunque se guarden en float64.
    float32_values: bool = True


# Referencias neuronales (`reference_run._evaluate`, lote de la campaña), Ridge y XGBoost
# (`tabular_corpus`, `external_corpus`, lote tabular), GRU candidata (`candidate_run`) y
# familia Titans (`titans_walk_forward`, también MARS-TITAN y CM-v1).
WRITERS = {
    "neural": Writer(pa.float32(), pa.float32(), False, 256),
    "ridge": Writer(pa.float64(), None, False, 1024, float32_values=False),
    "xgboost": Writer(pa.float32(), None, False, 1024),
    "episodic_gru": Writer(pa.float32(), pa.float32(), True, 65_536),
    "titans": Writer(pa.float64(), pa.float64(), True, 65_536),
}
# Escritor de cada modelo de `training.masked_campaign.EXECUTORS`.
MODEL_WRITERS = {
    "neural": "neural",
    "ridge": "ridge",
    "xgboost": "xgboost",
    "episodic_gru": "episodic_gru",
    "titans_mac": "titans",
    "mars_titan": "titans",
    "cm_v1_core": "titans",
    "cm_v1": "titans",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def view_assets(manifest_path):
    """Activos de una vista con su mercado y sus filas por tramo, sin abrir sus datos."""
    manifest, _ = read_manifest(Path(manifest_path), 64 * 1024**2)
    assets = manifest.get("assets")
    _require(
        isinstance(assets, list)
        and assets
        and all(
            isinstance(row, dict)
            and isinstance(row.get("market"), str)
            and isinstance(row.get("symbol"), str)
            and isinstance(row.get("counts"), dict)
            for row in assets
        ),
        "El manifiesto de la vista no declara sus activos y recuentos",
    )
    return [
        (f"{row['market']}/{row['symbol']}", row["market"], row["counts"]) for row in assets
    ], manifest["counts"]


def synthetic_keys(assets, partition, rng):
    """Claves y objetivo sintéticos en bloques por activo con las filas de cada activo.

    Cada activo ocupa un subconjunto ordenado de las sesiones del tramo, cuyo número es el
    mayor recuento de un activo. Las sesiones son días consecutivos a la hora de cierre de
    su mercado. El identificador de muestra repite el formato `mercado/símbolo/instante`.
    """
    rows = [(key, market, counts[partition]) for key, market, counts in assets]
    rows = [row for row in rows if row[2] > 0]
    _require(rows, f"El tramo {partition} no tiene filas")
    sessions = max(count for _, _, count in rows)
    base = int(np.datetime64("2010-01-04T00:00", "us").astype(np.int64))
    keys, markets, moments = [], [], []
    for key, market, count in rows:
        chosen = np.sort(rng.choice(sessions, count, replace=False))
        moments.append(base + chosen * _DAY + _CLOSE_HOURS.get(market, 12) * 3_600_000_000)
        keys.extend([key] * count)
        markets.extend([market] * count)
    moment = np.concatenate(moments).astype(np.int64)
    return pa.table(
        {
            "sample_id": pa.array(
                [f"{key}/{at}" for key, at in zip(keys, moment.tolist(), strict=True)],
                pa.string(),
            ),
            "asset_id": pa.array(keys, pa.string()),
            "market": pa.array(markets, pa.string()),
            "prediction_at": pa.array(moment, pa.timestamp("us", tz="UTC")),
            # El objetivo es común a todos los ajustes de la ventana, como en la vista.
            "target": pa.array(rng.standard_normal(len(moment)) * 0.02, pa.float64()),
        }
    )


def writer_table(keys, writer, rng):
    """Tabla con las columnas, tipos y orden de filas de un escritor.

    Los modelos calculan en float32 y la familia Titans guarda esos valores en float64.
    Ridge resuelve en float64, así que sus predicciones usan toda la mantisa.
    """
    count = keys.num_rows
    median = rng.standard_normal(count) * 0.003
    if writer.float32_values:
        median = median.astype(np.float32)
    columns = {name: keys[name] for name in (*KEY_COLUMNS, "target")}
    columns["prediction"] = pa.array(median.astype(writer.prediction.to_pandas_dtype()))
    columns["zero"] = pa.array(np.zeros(count), pa.float64())
    if writer.quantiles is not None:
        spread = np.abs(rng.standard_normal((count, 2)) * 0.004).astype(np.float32)
        levels = (
            median - spread[:, 0] - spread[:, 1],
            median - spread[:, 0],
            median,
            median + spread[:, 0],
            median + spread[:, 0] + spread[:, 1],
        )
        dtype = writer.quantiles.to_pandas_dtype()
        for name, values in zip(QUANTILE_COLUMNS, levels, strict=True):
            columns[name] = pa.array(values.astype(dtype))
    table = pa.table(columns)
    if writer.chronological:
        order = np.lexsort(
            (
                table["asset_id"].to_numpy(zero_copy_only=False),
                table["prediction_at"].cast(pa.int64()).to_numpy(),
            )
        )
        table = table.take(order)
    return table


def write_current(table, path, writer):
    """Repetir `atomic_parquet_batches`: un grupo por lote y zstd sin más opciones."""
    with pq.ParquetWriter(path, table.schema, compression="zstd") as stream:
        for start in range(0, table.num_rows, writer.row_group):
            stream.write_table(table.slice(start, writer.row_group))


def split_shared(table, rows, writer, *, keep_order=False):
    """Decimales propios de un ajuste en el orden común o con su índice de fila."""
    names = list(QUANTILE_COLUMNS) if writer.quantiles is not None else ["prediction"]
    if writer.quantiles is not None:
        _require(
            same_table(
                table.select(["prediction"]), table.select([MEDIAN]).rename_columns(["prediction"])
            ),
            "La predicción puntual no repite los bits de la mediana",
        )
    zero = table["zero"].to_numpy()
    _require(
        np.array_equal(zero.view(np.uint64), np.zeros(len(zero), np.uint64)),
        "El control nulo no vale cero positivo",
    )
    _require(
        same_table(canonical_rows(table), rows),
        "Las claves u objetivos no son los de la tabla común",
    )
    index = _row_index(table)
    if index is None or keep_order:
        columns = {name: table[name] for name in names}
        if index is not None:
            columns["row"] = pa.array(index, pa.int32())
        return pa.table(columns)
    return table.select(names).take(_canonical_order(table))


def _row_index(table):
    """Posición de cada fila en la tabla común, o nada si comparten orden."""
    order = _canonical_order(table)
    if np.array_equal(order, np.arange(len(order))):
        return None
    index = np.empty(len(order), dtype=np.int64)
    index[order] = np.arange(len(order))
    return index


def restore_shared(rows, outputs, schema):
    """Reconstruir la tabla de un ajuste desde la tabla común y sus decimales.

    Sin índice de fila, la tabla sale en el orden común.
    """
    if "row" in outputs.column_names:
        rows = rows.take(outputs["row"])
    count = rows.num_rows
    columns = {}
    for field in schema:
        if field.name in rows.column_names:
            columns[field.name] = rows[field.name]
        elif field.name == "zero":
            columns[field.name] = pa.array(np.zeros(count), field.type)
        elif field.name == "prediction" and "prediction" not in outputs.column_names:
            columns[field.name] = outputs[MEDIAN]
        else:
            columns[field.name] = outputs[field.name]
    return pa.table(columns, schema=schema)


def _size_and_check(path, expected):
    restored = pq.read_table(path, use_threads=False)
    _require(same_table(restored, expected), f"{path.name} no se relee con los mismos bits")
    return os.path.getsize(path)


def measure_partition(assets, partition, directory, *, seed=0, writers=None):
    """Bytes de cada escritor y disposición en un tramo, con la relectura comprobada.

    Devuelve las filas, los bytes de la tabla común y, por escritor, los bytes de cada
    disposición. Los archivos se escriben en `directory` y se borran al medirlos.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    keys = synthetic_keys(assets, partition, rng)
    result = dict(rows=keys.num_rows, writers={})
    shared_path = directory / f"{partition}-rows.parquet"
    rows = None
    for name in writers or WRITERS:
        writer = WRITERS[name]
        table = writer_table(keys, writer, rng)
        if rows is None:
            rows = canonical_rows(table)
            write_large(rows, shared_path)
            result["shared_rows_bytes"] = _size_and_check(shared_path, rows)
            shared_path.unlink()
        sizes = {}
        for layout in LAYOUTS:
            path = directory / f"{partition}-{name}-{layout}.parquet"
            if layout == "current":
                write_current(table, path, writer)
                sizes[layout] = _size_and_check(path, table)
            elif layout == "large_groups":
                write_large(table, path)
                sizes[layout] = _size_and_check(path, table)
            else:
                ordered = layout == "shared_rows_ordered"
                outputs = split_shared(table, rows, writer, keep_order=ordered)
                write_large(outputs, path)
                restored = restore_shared(rows, pq.read_table(path), table.schema)
                expected = table if ordered else table.take(_canonical_order(table))
                _require(
                    same_table(restored, expected), f"{name} no se reconstruye con los mismos bits"
                )
                sizes[layout] = _size_and_check(path, outputs)
            path.unlink()
        result["writers"][name] = sizes
    return result


def adapter_table(keys, rng):
    """Predicciones reservadas de la etapa de adaptadores (`posttraining.heldout`).

    El centro y los cuantiles del adaptador se calculan en float64. La predicción y el
    centro repiten los bits de la mediana y el padre conserva valores de float32.
    """
    count = keys.num_rows
    median = rng.standard_normal(count) * 0.003
    spread = np.abs(rng.standard_normal((count, 2)) * 0.004)
    columns = {name: keys[name] for name in (*KEY_COLUMNS, "target")}
    columns.update(
        prediction=pa.array(median),
        parent=pa.array((rng.standard_normal(count) * 0.003).astype(np.float32).astype(float)),
        zero=pa.array(np.zeros(count)),
        center=pa.array(median),
    )
    levels = (
        median - spread[:, 0] - spread[:, 1],
        median - spread[:, 0],
        median,
        median + spread[:, 0],
        median + spread[:, 0] + spread[:, 1],
    )
    columns.update(zip(QUANTILE_COLUMNS, (pa.array(v) for v in levels), strict=True))
    return pa.table(columns)


def adapter_outputs(table):
    """Decimales propios de un adaptador sobre la tabla común: cuantiles y padre en float32.

    La predicción y el centro repiten los bits de la mediana y el control nulo vale cero
    positivo, así que se reconstruyen. El padre conserva valores de float32 y se guarda en
    ese tipo después de comprobar que vuelve a float64 con los mismos bits.
    """
    for name in ("prediction", "center"):
        _require(
            same_table(table.select([name]), table.select([MEDIAN]).rename_columns([name])),
            f"La columna {name} no repite los bits de la mediana",
        )
    zero = table["zero"].to_numpy()
    _require(
        np.array_equal(zero.view(np.uint64), np.zeros(len(zero), np.uint64)),
        "El control nulo no vale cero positivo",
    )
    parent = table["parent"].cast(pa.float32())
    _require(
        same_table(pa.table({"p": parent.cast(pa.float64())}), pa.table({"p": table["parent"]})),
        "El padre no conserva valores de float32",
    )
    columns = {name: table[name] for name in QUANTILE_COLUMNS}
    return pa.table(dict(columns, parent=parent))


STRATA = 4
ALL_ROWS = STRATA


def session_aggregates(table, rng, *, quantiles):
    """Agregados por mercado, sesión y estrato que sustituirían a las filas de un ajuste.

    Los estratos son las cuatro combinaciones de presencia de noticias y fundamentales,
    aquí asignadas al azar, y un estrato adicional con todas las filas de la sesión. Cada
    fila conserva sumas y recuentos, no medias, para que la comparación pueda agregarlas
    por bloques. La correlación de rangos solo existe en el estrato con todas las filas.
    """
    market = table["market"].to_numpy(zero_copy_only=False).astype(str)
    moment = table["prediction_at"].cast(pa.int64()).to_numpy()
    target = table["target"].to_numpy()
    point = table["prediction"].to_numpy().astype(np.float64)
    stratum = rng.integers(0, STRATA, len(target))
    keys = np.concatenate([stratum, np.full(len(target), ALL_ROWS)])
    market, moment = np.concatenate([market, market]), np.concatenate([moment, moment])
    target, point = np.concatenate([target, target]), np.concatenate([point, point])
    groups, inverse = np.unique(np.rec.fromarrays([market, moment, keys]), return_inverse=True)

    def total(values):
        return np.bincount(inverse, weights=values, minlength=len(groups))

    def count(mask):
        return np.bincount(inverse, weights=mask.astype(float), minlength=len(groups)).astype(
            np.int64
        )

    error = point - target
    up, rise = point > 0, target > 0
    columns = dict(
        market=pa.array(groups["f0"].astype(str)),
        prediction_at=pa.array(groups["f1"], pa.timestamp("us", tz="UTC")),
        stratum=pa.array(groups["f2"].astype(np.int8)),
        samples=pa.array(count(np.ones(len(target), bool))),
        absolute_error=pa.array(total(np.abs(error))),
        squared_error=pa.array(total(error**2)),
        zero_absolute_error=pa.array(total(np.abs(target))),
        zero_squared_error=pa.array(total(target**2)),
        up_rise=pa.array(count(up & rise)),
        up_fall=pa.array(count(up & ~rise)),
        down_rise=pa.array(count(~up & rise)),
        down_fall=pa.array(count(~up & ~rise)),
    )
    if quantiles:
        levels = np.stack([table[name].to_numpy() for name in QUANTILE_COLUMNS], axis=1)
        levels = np.concatenate([levels, levels]).astype(np.float64)
        for tau, name in zip((0.025, 0.1, 0.5, 0.9, 0.975), QUANTILE_COLUMNS, strict=True):
            gap = target - levels[:, QUANTILE_COLUMNS.index(name)]
            columns[f"pinball_{name[-4:]}"] = pa.array(
                total(np.maximum(tau * gap, (tau - 1) * gap))
            )
        columns["inside_80"] = pa.array(count((target >= levels[:, 1]) & (target <= levels[:, 3])))
        columns["inside_95"] = pa.array(count((target >= levels[:, 0]) & (target <= levels[:, 4])))
    rank = rng.uniform(-1, 1, len(groups))
    columns["rank_ic"] = pa.array(rank, mask=groups["f2"] != ALL_ROWS)
    return pa.table(columns)


def index_events(keys):
    """Eventos del índice de observaciones: una muestra y una etiqueta madura por fila.

    Repite las columnas de `memory.financial_observations._records`. La etiqueta madura en
    la sesión siguiente y cada activo usa su posición en el orden de mercado y símbolo.
    """
    asset = keys["asset_id"].to_numpy(zero_copy_only=False).astype(str)
    names, identity = np.unique(asset, return_inverse=True)
    moment = keys["prediction_at"].cast(pa.int64()).to_numpy()
    starts = np.r_[0, np.flatnonzero(asset[1:] != asset[:-1]) + 1]
    row = np.arange(len(asset)) - np.repeat(starts, np.diff(np.r_[starts, len(asset)]))
    samples = dict(
        event_at=moment,
        kind=np.zeros(len(moment), np.int64),
        asset_id=identity.astype(np.int64),
        source_group=row // 4096,
        source_row=row,
        sample_at=moment,
    )
    labels = dict(samples, event_at=moment + _DAY, kind=np.ones(len(moment), np.int64))
    labels["source_group"] = np.full(len(moment), -1, np.int64)
    return pa.concat_tables([pa.table(samples), pa.table(labels)])


def index_bytes(keys, directory):
    """Bytes del índice sin ordenar y ordenado, con las opciones de escritura del código.

    El archivo sin ordenar repite `atomic_parquet_batches` con un grupo por tabla de activo
    y el ordenado, el `COPY` de DuckDB con grupos de 2048 filas y zstd.
    """
    import duckdb

    events = index_events(keys)
    raw = Path(directory) / "index-raw.parquet"
    asset = events["asset_id"].to_numpy()
    order = np.lexsort((events["kind"].to_numpy(), asset))
    blocked = events.take(order)
    bounds = np.r_[0, np.flatnonzero(np.diff(blocked["asset_id"].to_numpy())) + 1, len(order)]
    with pq.ParquetWriter(raw, blocked.schema, compression="zstd") as stream:
        for start, end in zip(bounds[:-1], bounds[1:], strict=True):
            stream.write_table(blocked.slice(start, end - start))
    raw_size = os.path.getsize(raw)
    raw.unlink()
    path = Path(directory) / "index-events.parquet"
    with duckdb.connect(config={"threads": 2, "memory_limit": "2GiB"}) as connection:
        connection.register("events", events)
        connection.execute(
            "COPY (SELECT * FROM events ORDER BY event_at, kind, asset_id, sample_at) "
            f"TO '{path}' (FORMAT PARQUET, ROW_GROUP_SIZE 2048, COMPRESSION ZSTD)"
        )
    size = os.path.getsize(path)
    path.unlink()
    return dict(rows=events.num_rows, bytes=size, raw_bytes=raw_size)


def measure_extras(assets, partition, directory, *, seed=0):
    """Adaptadores, agregados por sesión e índice de observaciones de un tramo."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    keys = synthetic_keys(assets, partition, rng)
    result = dict(rows=keys.num_rows)
    adapter = adapter_table(keys, rng)
    for layout in ("current", "large_groups"):
        path = directory / f"{partition}-adapter-{layout}.parquet"
        if layout == "current":
            write_current(adapter, path, Writer(pa.float64(), pa.float64(), False, 256))
        else:
            write_large(adapter, path)
        result[f"adapter_{layout}"] = _size_and_check(path, adapter)
        path.unlink()
    own = adapter_outputs(adapter)
    path = directory / f"{partition}-adapter-shared.parquet"
    write_large(own, path)
    result["adapter_shared_rows"] = _size_and_check(path, own)
    path.unlink()
    for name, quantiles in (("neural", True), ("ridge", False)):
        table = writer_table(keys, WRITERS[name], rng)
        aggregates = session_aggregates(table, rng, quantiles=quantiles)
        path = directory / f"{partition}-aggregates-{name}.parquet"
        write_large(aggregates, path)
        result[f"aggregates_{'quantile' if quantiles else 'point'}"] = _size_and_check(
            path, aggregates
        )
        result["aggregate_rows"] = aggregates.num_rows
        path.unlink()
    result["index"] = index_bytes(keys, directory)
    return result


def tape_bytes(sessions, assets, directory, *, seed=0):
    """Bytes de una cinta de política con el esquema y los grupos de `simulation.storage`."""
    rng = np.random.default_rng(seed)
    rows = sessions * assets
    base = int(np.datetime64("2010-01-04T20:00", "us").astype(np.int64))
    close = base + np.arange(sessions, dtype=np.int64) * _DAY
    prices = np.exp(np.cumsum(rng.standard_normal((sessions, assets)) * 0.02, axis=0)) * 50
    columns = {
        name: prices.reshape(-1) * (1 + rng.standard_normal(rows) * 0.001)
        for name in ("open", "high", "low", "close")
    }
    columns.update(
        volume=np.round(rng.lognormal(12, 1, rows)),
        score=(rng.standard_normal(rows) * 0.003).astype(np.float32).astype(float),
        close_time=np.repeat(close, assets),
        open_time=np.repeat(close - 23_400_000_000, assets),
        prediction_time=np.repeat(close - _DAY, assets),
        asset=np.tile(np.asarray([f"A{i:04d}" for i in range(assets)]), sessions),
    )
    table = pa.table(columns)
    path = Path(directory) / "tape.parquet"
    pq.write_table(table, path, compression="zstd", row_group_size=assets * 16)
    size = _size_and_check(path, table)
    path.unlink()
    return size


def measure_view(manifest_path, directory, *, seed=0, writers=None):
    """Medir los tres tramos reservados de una vista y comprobar sus recuentos."""
    assets, counts = view_assets(manifest_path)
    result = {}
    for offset, partition in enumerate(HELD_OUT):
        measured = measure_partition(
            assets, partition, directory, seed=seed + offset, writers=writers
        )
        _require(
            measured["rows"] == counts[partition],
            f"Los activos de la vista no suman las filas de {partition}",
        )
        result[partition] = measured
    return result


# Valores que no se miden aquí, con su procedencia. El corpus ordenado de los adaptadores
# repite la medida de `docs/engineering/corpus-temporary-storage.md` (1.392.366.107 y
# 2.060.963.783 bytes para 321.610 filas de entrenamiento). La caché de cada padre es el
# presupuesto de `posttraining.parent_cache`. Los estados de adaptadores son la cota del inventario
# de `posttraining.run` (padre completo más adaptador, hasta 1,8 MB).
ADAPTER_DECLARED = dict(
    ordered_row_bytes=4330,
    input_row_bytes=6409,
    parent_cache_bytes=64 * 1024**2 + 100 * 1024,
    state_bytes=1_800_000,
    retained_states=3,
    job_report_bytes=3 * 100 * 1024,
)


ADAPTER_LAYOUTS = ("current", "large_groups", "shared_rows")


def measure_extras_report(views, directory, *, tape_sessions=252, tape_assets=128):
    """Medidas fuera del ajuste base sobre la ventana más poblada de cada ámbito."""
    adapter, aggregate, index, raw = [], [], [], []
    measured = {}
    for scope, folder in views.items():
        report, _ = read_manifest(Path(folder) / "report.json", 1024**2)
        largest = max(report["folds"], key=lambda row: row["counts"]["train"])["id"]
        assets, _ = view_assets(Path(folder) / largest / "manifest.json")
        for offset, partition in enumerate(("evaluation", "train")):
            result = measure_extras(assets, partition, directory, seed=offset)
            rows = result["rows"]
            measured[f"{scope}/{largest}/{partition}"] = result
            adapter.append(tuple(result[f"adapter_{name}"] / rows for name in ADAPTER_LAYOUTS))
            aggregate.append(result["aggregates_quantile"] / rows)
            index.append(result["index"]["bytes"] / result["index"]["rows"])
            raw.append(result["index"]["raw_bytes"] / result["index"]["rows"])
    tape = tape_bytes(tape_sessions, tape_assets, directory)
    return dict(
        schema_version=1,
        kind="historical_masked_campaign_storage_extras",
        aggregate_bytes_per_row=max(aggregate),
        index_bytes_per_row=max(index),
        index_build_bytes_per_row=max(raw),
        tape_bytes=tape,
        tape_shape=dict(sessions=tape_sessions, assets=tape_assets),
        adapter=dict(
            ADAPTER_DECLARED,
            **{
                f"row_bytes_{name}": max(row[i] for row in adapter)
                for i, name in enumerate(ADAPTER_LAYOUTS)
            },
        ),
        measured=measured,
        pyarrow=pa.__version__,
    )


def main(argv=None):
    import argparse

    from mars_titan.data.storage import atomic_json

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    view = commands.add_parser("view", help="Medir los tramos reservados de una vista")
    view.add_argument("--manifest", type=Path, required=True)
    extras = commands.add_parser("extras", help="Adaptadores, agregados, índices y cintas")
    extras.add_argument("--views", action="append", required=True)
    extras.add_argument("--output", type=Path, required=True)
    for command in (view, extras):
        command.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "view":
        print(json.dumps(measure_view(args.manifest, args.scratch), indent=2))
        return 0
    views = dict(value.partition("=")[::2] for value in args.views)
    result = measure_extras_report(views, args.scratch)
    atomic_json(args.output, result)
    print(json.dumps({k: v for k, v in result.items() if k != "measured"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
