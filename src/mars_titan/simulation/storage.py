"""Precios y predicciones en Parquet, ligados a un manifiesto confirmado al final."""

import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from mars_titan.data.cohort_files import read_manifest, safe_destination
from mars_titan.data.storage import atomic_json, sha256

from .market import MAX_TAPE_CELLS, MarketTape
from .portfolio import MAX_INSTRUMENTS, CorporateAction, Instrument, Period

PRICE_COLUMNS = ("open", "high", "low", "close", "volume")
MAX_BYTES = 256 * 1024**2


def _instrument(identity):
    """Reconstruir un Instrument desde su identidad y comprobar que la conserva."""
    values = dict(identity)
    for key in ("price_limits", "taxes"):
        if key in values:
            values[key] = tuple(Period(**period) for period in values[key])
    instrument = Instrument(**values)
    if instrument.identity() != identity:
        raise ValueError("Las reglas de mercado del escenario no conservan su identidad")
    return instrument


def instruments_from_manifest(manifest, assets):
    """Reglas de mercado de un manifiesto de versión 2, o None en la versión 1."""
    rules = manifest.get("instruments")
    if (manifest.get("schema_version") == 2) != (rules is not None):
        raise ValueError("El manifiesto declara reglas de mercado sin su versión")
    if rules is None:
        return None
    if not isinstance(rules, dict) or set(rules) != set(assets):
        raise ValueError("Las reglas de mercado deben cubrir cada activo")
    try:
        return {asset: _instrument(rules[asset]) for asset in assets}
    except TypeError as error:
        raise ValueError("Las reglas de mercado del escenario no tienen su contrato") from error


def write_tape(tape, destination, *, instruments=None):
    """Escribir la cinta. Con ``instruments`` el manifiesto pasa a la versión 2 con sus reglas."""
    rules = None
    if instruments is not None:
        if (
            not isinstance(instruments, dict)
            or set(instruments) != set(tape.assets)
            or any(
                not isinstance(item, Instrument) or item.currency != tape.currency
                for item in instruments.values()
            )
        ):
            raise ValueError("Las reglas declaradas deben cubrir cada activo en su moneda")
        rules = {asset: instruments[asset].identity() for asset in tape.assets}
    destination = Path(destination)
    safe_destination(destination)
    destination.mkdir(parents=True, exist_ok=False)
    count = len(tape.assets)
    columns = {name: tape.prices[:, :, i].reshape(-1) for i, name in enumerate(PRICE_COLUMNS)}
    columns.update(
        score=tape.scores.reshape(-1),
        close_time=np.repeat(tape.close_times, count),
        open_time=np.repeat(tape.open_times, count),
        prediction_time=np.repeat(tape.prediction_times, count),
        asset=np.tile(np.asarray(tape.assets), len(tape)),
    )
    pending, path = destination / ".market.pending", destination / "market.parquet"
    pq.write_table(pa.table(columns), pending, compression="zstd", row_group_size=count * 16)
    with pending.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(pending, path)
    manifest = dict(
        schema_version=1 if rules is None else 2,
        identity=tape.identity,
        actions=[asdict(action) for action in tape.actions],
        tape_sha256=tape.sha256,
        file_sha256=sha256(path),
        file_bytes=path.stat().st_size,
        sessions=len(tape),
        assets=len(tape.assets),
        final_test_opened=False,
    )
    if rules is not None:
        manifest["instruments"] = rules
    atomic_json(destination / "manifest.json", manifest)


def read_tape(directory):
    directory = Path(directory)
    safe_destination(directory)
    manifest, _ = read_manifest(directory / "manifest.json", 4 * 1024**2)
    path = directory / "market.parquet"
    safe_destination(path)
    count, sessions = manifest["assets"], manifest["sessions"]
    if (
        manifest.get("schema_version") not in (1, 2)
        or manifest.get("final_test_opened") is not False
        or type(count) is not int
        or not 1 <= count <= MAX_INSTRUMENTS
        or type(sessions) is not int
        or not 2 <= sessions <= 8192
        or count * sessions > MAX_TAPE_CELLS
        or not path.is_file()
        or not 0 < path.stat().st_size <= MAX_BYTES
        or path.stat().st_size != manifest["file_bytes"]
        or sha256(path) != manifest["file_sha256"]
    ):
        raise ValueError("El escenario no conserva su integridad o presupuesto")
    parquet = pq.ParquetFile(path)
    selected = (*PRICE_COLUMNS, "score", "close_time", "open_time", "prediction_time", "asset")
    if parquet.metadata.num_rows != count * sessions or set(parquet.schema.names) != set(selected):
        raise ValueError("El Parquet no conserva las dimensiones del escenario")
    if (
        sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.num_row_groups))
        > MAX_BYTES
    ):
        raise ValueError("El Parquet descomprimido excede el presupuesto")
    table = parquet.read(columns=selected, use_threads=False)
    values = {
        name: table[name].to_numpy(zero_copy_only=False).reshape(sessions, count)
        for name in selected
    }
    identity = manifest["identity"]
    instruments_from_manifest(manifest, identity["assets"])
    if not np.all(values["asset"] == np.asarray(identity["assets"])[None]):
        raise ValueError("El orden de los activos no corresponde a su manifiesto")
    for name in ("close_time", "open_time", "prediction_time"):
        if not np.all(values[name] == values[name][:, :1]):
            raise ValueError("Las filas de una sesión no comparten calendario")
    result = MarketTape(
        np.stack([values[name] for name in PRICE_COLUMNS], axis=-1),
        values["close_time"][:, 0],
        identity["assets"],
        values["score"],
        domain=identity["domain"],
        currency=identity["currency"],
        partition=identity["partition"],
        prediction_times=values["prediction_time"][:, 0],
        open_times=values["open_time"][:, 0],
        parent_id=identity["parent_id"],
        audit=identity["audit"],
        source_identity=identity["source"],
        actions=[CorporateAction(**value) for value in manifest["actions"]],
    )
    if result.sha256 != manifest["tape_sha256"]:
        raise ValueError("El escenario reconstruido no conserva su integridad")
    return result
