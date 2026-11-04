"""Puntuar archivos de desarrollo por sesiones completas con memoria de lectura acotada."""

import re
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import safe_destination
from mars_titan.environments.cohorts import FINAL_TEST_START_US, MAX_COHORT_ASSETS

from .forecast_panel import ForecastPanel
from .forecast_scores import SessionScores, score_sessions

BASE_COLUMNS = ("sample_id", "market", "prediction_at", "target")
MAX_SESSIONS = 50_000


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _name(value):
    return type(value) is str and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,127}", value) is not None


def check_prediction_schema(path, rows, columns):
    """Comprobar pie, filas y tipos antes de ordenar un Parquet de pronósticos."""
    path = Path(path)
    safe_destination(path)
    _require(path.is_file() and path.stat().st_size >= 12, "Falta un pie Parquet completo")
    with path.open("rb") as stream:
        stream.seek(-8, 2)
        footer = stream.read(8)
    _require(
        footer[4:] == b"PAR1" and int.from_bytes(footer[:4], "little") <= 64 * 1024**2,
        "El pie Parquet no es válido o supera 64 MiB",
    )
    parquet = pq.ParquetFile(path)
    schema = parquet.schema_arrow
    _require(
        type(rows) is int
        and 1 <= rows <= MAX_SESSIONS * MAX_COHORT_ASSETS
        and rows == parquet.metadata.num_rows,
        "El Parquet no conserva su número de filas dentro del presupuesto",
    )
    _require(
        len(schema.names) == len(set(schema.names)) and set(columns) <= set(schema.names),
        "El archivo no conserva las columnas de predicción",
    )
    for name in columns:
        dtype = schema.field(name).type
        valid = (
            dtype == pa.timestamp("us", tz="UTC")
            if name == "prediction_at"
            else pa.types.is_string(dtype) or pa.types.is_large_string(dtype)
            if name in ("sample_id", "market")
            else pa.types.is_float64(dtype)
        )
        _require(valid, f"El tipo de {name} no conserva el contrato de pronósticos")
    return schema


def _columns(points, equal_points, quantiles):
    _require(
        type(points) is dict
        and 1 <= len(points) <= 32
        and all(_name(k) and (v is None or _name(v)) for k, v in points.items()),
        "Las predicciones necesitan nombres y columnas explícitos y acotados",
    )
    _require(
        type(equal_points) in (tuple, list)
        and len(equal_points) <= 32
        and all(
            type(pair) in (tuple, list) and len(pair) == 2 and all(map(_name, pair))
            for pair in equal_points
        ),
        "Las igualdades de emisión necesitan dos columnas válidas",
    )
    _require(
        type(quantiles) is dict and len(quantiles) <= 1 and set(quantiles) <= set(points),
        "Los cuantiles deben pertenecer a una predicción emitida",
    )
    columns = [*BASE_COLUMNS, *(v for v in points.values() if v is not None)]
    columns.extend(name for pair in equal_points for name in pair)
    for name, specification in quantiles.items():
        _require(
            points[name] is not None
            and type(specification) in (tuple, list)
            and len(specification) == 2,
            "Los cuantiles necesitan columnas y niveles propios",
        )
        names, levels = specification
        _require(
            type(names) in (tuple, list)
            and type(levels) in (tuple, list)
            and 1 <= len(names) == len(levels) <= 32
            and all(map(_name, names)),
            "Las columnas y niveles de cuantiles deben estar completos",
        )
        _require(len(set(names)) == len(names), "Una columna de cuantiles está repetida")
        columns.extend(names)
    return tuple(dict.fromkeys(columns))


def _sessions(reader):
    """Reunir solo la sesión actual aunque cruce lotes Arrow y grupos Parquet."""
    pending, size, previous = [], 0, None
    for batch in reader:
        table = pa.Table.from_batches([batch])
        position = table.column_names.index("prediction_at")
        table = table.set_column(
            position, "prediction_at", table["prediction_at"].cast(pa.timestamp("us", tz="UTC"))
        )
        _require(
            table["market"].null_count == table["prediction_at"].null_count == 0,
            "Una sesión carece de mercado o instante",
        )
        markets = table["market"].to_numpy()
        times = table["prediction_at"].cast(pa.int64()).to_numpy()
        starts = np.r_[
            0, np.flatnonzero((markets[1:] != markets[:-1]) | (times[1:] != times[:-1])) + 1
        ]
        for start, end in zip(starts, np.r_[starts[1:], len(table)], strict=True):
            key = (markets[start], times[start])
            if pending and previous != key:
                yield pa.concat_tables(pending)
                pending, size = [], 0
            size += int(end - start)
            _require(size <= MAX_COHORT_ASSETS, "La sesión excede el límite de activos")
            pending.append(table.slice(int(start), int(end - start)))
            previous = key
    if pending:
        yield pa.concat_tables(pending)


def _session_scores(table, points, equal_points, quantiles, minimum):
    _require(all(c.null_count == 0 for c in table.columns), "Hay pronósticos ausentes")
    for first, second in equal_points:
        _require(
            np.array_equal(table[first].to_numpy(), table[second].to_numpy()),
            "La condición de control no coincide con la emisión original",
        )
    result = {}
    for name, column in points.items():
        specification = quantiles.get(name)
        panel = ForecastPanel.from_columns(
            table["sample_id"],
            table["market"].to_numpy(),
            table["prediction_at"].to_numpy(),
            table["target"].to_numpy(),
            np.zeros(len(table)) if column is None else table[column].to_numpy(),
            quantiles=None
            if specification is None
            else np.column_stack([table[c].to_numpy() for c in specification[0]]),
            levels=None if specification is None else specification[1],
        )
        with np.errstate(over="ignore", invalid="ignore"):
            _require(
                np.isfinite(np.square(panel.prediction - panel.target)).all(),
                "Los errores no admiten MSE finito",
            )
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                result[name] = score_sessions(panel, rank_ic_min_assets=minimum)
        except FloatingPointError as error:
            raise ValueError("Un estadístico de pronósticos desborda su cálculo finito") from error
    return result


def score_prediction_file(
    path,
    points,
    *,
    rows,
    minimum,
    temporary,
    batch_size=4096,
    max_sessions=MAX_SESSIONS,
    equal_points=(),
    quantiles=None,
):
    """Devolver puntuaciones e inventario de recursos, sin seleccionar ni ajustar modelos.

    ``points`` asigna columnas a nombres de condiciones. None genera el predictor
    residual cero. Los cuantiles son opcionales y pertenecen a una salida existente.
    El llamador verifica las huellas de fuentes y aplica su guarda científica.
    """
    _require(
        type(batch_size) is int
        and 1 <= batch_size <= 8192
        and type(minimum) is int
        and 3 <= minimum <= MAX_COHORT_ASSETS
        and type(max_sessions) is int
        and 1 <= max_sessions <= MAX_SESSIONS,
        "El lote, mínimo de activos o presupuesto de sesiones no son válidos",
    )
    quantiles = {} if quantiles is None else quantiles
    columns = _columns(points, equal_points, quantiles)
    path, temporary = Path(path), Path(temporary)
    check_prediction_schema(path, rows, columns)
    parts, chunk = {name: [] for name in points}, []
    count, maximum = 0, 0

    def flush():
        for name in parts:
            parts[name].append(SessionScores.concatenate([s[name] for s in chunk]))
        chunk.clear()

    config = dict(
        threads=2,
        memory_limit="256MiB",
        temp_directory=str(temporary / "spill"),
        max_temp_directory_size="1GiB",
        autoinstall_known_extensions=False,
        autoload_known_extensions=False,
    )
    with duckdb.connect(config=config) as connection:
        actual, unique, first, last = connection.execute(
            "SELECT count(*),count(DISTINCT sample_id),min(epoch_us(prediction_at)),"
            "max(epoch_us(prediction_at)) FROM read_parquet(?)",
            [str(path)],
        ).fetchone()
        _require(
            actual == rows == unique
            and first is not None
            and 0 <= first <= last < FINAL_TEST_START_US,
            "Hay identidades repetidas o instantes fuera del desarrollo",
        )
        selected = ",".join(f'"{name}"' for name in columns)
        reader = connection.execute(
            f"SELECT {selected} FROM read_parquet(?) ORDER BY market,prediction_at,sample_id",
            [str(path)],
        ).to_arrow_reader(batch_size=batch_size)
        for table in _sessions(reader):
            count += 1
            maximum = max(maximum, len(table))
            _require(count <= max_sessions, "Se excede el presupuesto de sesiones")
            chunk.append(_session_scores(table, points, equal_points, quantiles, minimum))
            if len(chunk) == 128:
                flush()
    if chunk:
        flush()
    return {name: SessionScores.concatenate(values) for name, values in parts.items()}, dict(
        rows=rows,
        sessions=count,
        largest_session_rows=maximum,
        arrow_batch_rows=batch_size,
        duckdb_threads=2,
        duckdb_memory="256MiB",
        duckdb_spill_limit="1GiB",
        max_sessions=max_sessions,
        cohort_hash="hierarchical_session_concatenation_v1",
    )
