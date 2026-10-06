"""Componer las versiones publicadas de STLFSI sin modificar sus niveles ni las fuentes."""

import csv
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .batches import atomic_parquet_batches
from .macro import _available
from .macro_coverage import _check_schema, _publish_directory, _read_catalog, _selected_batches
from .macro_model_vintages import (
    STRESS_HISTORY_POLICY,
    model_vintage_contract,
    stress_history_identity,
)
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_TARGET = "us_financial_stress"
_V3 = "us_financial_stress_v3"


def _receipt(panel, report_path, catalog, *, market, start, end):
    for path in (panel, report_path, catalog):
        if path.is_symlink() or not path.is_file():
            raise ValueError("La fuente debe ser un archivo regular, sin enlaces")
    with report_path.open("rb") as stream:
        content = stream.read(1024**2 + 1)
    if len(content) > 1024**2:
        raise ValueError("El recibo de origen supera un MiB")
    report = json.loads(content)
    panel_hash, catalog_hash = sha256(panel), sha256(catalog)
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or report["schema_version"] not in {1, 2}
        or report.get("sha256", report.get("output_sha256")) != panel_hash
        or "output_sha256" in report
        and report["output_sha256"] != panel_hash
        or report.get("catalog_sha256") != catalog_hash
        or report.get("market") != market
        or type(report.get("rows")) is not int
        or not 0 < report["rows"] <= 8_000_000
    ):
        raise ValueError("El recibo no acredita el panel, el catálogo o el mercado de origen")
    try:
        first, last = (date.fromisoformat(report[key]) for key in ("start", "end"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("El recibo no acredita el periodo de origen") from error
    if first > start or last < end or last >= date(2024, 1, 1):
        raise ValueError("La cobertura del recibo no acredita el periodo anterior a 2024")
    size = panel.stat().st_size
    if size < 12:
        raise ValueError("El panel no tiene un pie Parquet completo")
    with panel.open("rb") as stream:
        stream.seek(-8, 2)
        footer = stream.read(8)
    if footer[4:] != b"PAR1" or int.from_bytes(footer[:4], "little") > min(8 * 1024**2, size - 12):
        raise ValueError("Los metadatos Parquet no cumplen el presupuesto")
    return report, {
        "panel_sha256": panel_hash,
        "report_sha256": hashlib.sha256(content).hexdigest(),
        "catalog_sha256": catalog_hash,
    }


def _validate_row(row, earliest):
    hashes = row["source_hashes"]
    if not isinstance(hashes, list) or any(
        not isinstance(value, str) or not re.fullmatch("[0-9a-fA-F]{64}", value) for value in hashes
    ):
        raise ValueError("La procedencia de una versión contiene huellas no válidas")
    available = row["available_at"]
    if available is not None and available > row["prediction_at"]:
        raise ValueError("La versión utiliza disponibilidad futura")
    if row["value"] is None:
        if not row["missing_reason"]:
            raise ValueError("Una ausencia necesita un motivo explícito")
        return
    try:
        period = date.fromisoformat(row["period_start"])
    except (TypeError, ValueError) as error:
        raise ValueError("La versión contiene un periodo no válido") from error
    if (
        not math.isfinite(row["value"])
        or row["missing_reason"] is not None
        or not hashes
        or available is None
        or available < earliest
        or period > available.date()
        or row["unit"] not in {"Index", "index"}
        or row["seasonal_adjustment"] != "Not Seasonally Adjusted"
    ):
        raise ValueError("La versión no acredita valor, unidad, procedencia o publicación")


def _source_rows(panel, report, entries, identifier, decisions, first, stop, earliest):
    moments, ids = pa.array(decisions), pa.array(sorted(entries))
    counts = np.zeros((len(decisions), len(entries)), dtype=np.uint16)
    rows, previous, schema, selected_bytes = {}, None, None, 0
    with pq.ParquetFile(panel) as file:
        _check_schema(file.schema_arrow)
        if file.metadata.num_rows != report["rows"]:
            raise ValueError("El recuento del panel contradice el recibo")
        time_index = next(
            i for i in range(len(file.schema)) if file.schema.column(i).path == "prediction_at"
        )
        for number in range(file.num_row_groups):
            stats = file.metadata.row_group(number).column(time_index).statistics
            if (
                not stats
                or not stats.has_min_max
                or stats.null_count
                or stats.min.date().isoformat() < report["start"]
                or stats.max.date().isoformat() > report["end"]
            ):
                raise ValueError("Las fechas del panel no acreditan el periodo del recibo")
        for batch, _, _ in _selected_batches(file, first, stop):
            schema = batch.schema
            d = pc.index_in(batch["prediction_at"], value_set=moments)
            i = pc.index_in(batch["indicator_id"], value_set=ids)
            if d.null_count or i.null_count:
                raise ValueError("El panel contiene una decisión o un indicador ajeno al contrato")
            times = batch["prediction_at"].to_numpy(zero_copy_only=False)
            if np.any(times[1:] < times[:-1]) or previous is not None and times[0] < previous:
                raise ValueError("Las decisiones del panel no están ordenadas")
            previous = times[-1]
            np.add.at(counts, (d.to_numpy(), i.to_numpy()), 1)
            if counts.max() > 1:
                raise ValueError("Hay un indicador duplicado en la misma decisión")
            selected = batch.filter(pc.equal(batch["indicator_id"], identifier))
            selected_bytes += selected.nbytes
            if (
                selected_bytes > 16 * 1024**2
                or (pc.max(pc.list_value_length(selected["source_hashes"])).as_py() or 0) > 256
            ):
                raise ValueError("La columna seleccionada supera el presupuesto de lectura")
            for row in selected.to_pylist():
                _validate_row(row, earliest)
                rows[row["prediction_at"]] = row
    if np.any(counts != 1):
        raise ValueError("Faltan filas del catálogo en el panel de origen")
    return rows, schema


def compose_stress_history(
    base: Path,
    base_report: Path,
    base_catalog: Path,
    v3: Path,
    v3_report: Path,
    v3_catalog: Path,
    output: Path,
    *,
    market: str,
    start: str,
    end: str,
) -> dict:
    """Publicar una columna y su catálogo separado, con los tramos STLFSI3 y STLFSI4."""
    base, base_report, base_catalog, v3, v3_report, v3_catalog, output = map(
        Path, (base, base_report, base_catalog, v3, v3_report, v3_catalog, output)
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError("La historia de versiones ya existe")
    first_day, last_day = date.fromisoformat(start), date.fromisoformat(end)
    if (
        first_day > last_day
        or last_day >= date(2024, 1, 1)
        or (last_day - first_day).days > 366 * 50
    ):
        raise ValueError("El periodo debe estar ordenado y mantener cerrada la reserva de 2024")
    sources = (base, base_report, base_catalog, v3, v3_report, v3_catalog)
    if len({path.resolve() for path in sources}) != len(sources):
        raise ValueError("Cada entrada necesita su propia identidad de origen")
    for source in (*sources, Path("dataset")):
        outside_source(source, output)
    receipts, identities = [], []
    for paths in ((base, base_report, base_catalog), (v3, v3_report, v3_catalog)):
        receipt, identity = _receipt(*paths, market=market, start=first_day, end=last_day)
        receipts.append(receipt)
        identities.append(identity)
    original, earlier = _read_catalog(base_catalog), _read_catalog(v3_catalog)
    if (
        len(original) != 140
        or _TARGET not in original
        or set(earlier) != {_V3}
        or model_vintage_contract(original[_TARGET]) != ("2022-11-10", None)
        or model_vintage_contract(earlier[_V3]) != ("2022-01-13", None)
        or any(_TARGET in entry["input_ids"].split("|") for entry in original.values())
    ):
        raise ValueError("Los catálogos no acreditan STLFSI3, STLFSI4 o la ausencia de derivados")
    clock = MarketClock(
        market,
        min(first_day, date(2022, 1, 1)).isoformat(),
        max(last_day + timedelta(days=7), date(2022, 11, 18)).isoformat(),
    )
    first = datetime.combine(first_day, datetime.min.time(), UTC)
    stop = datetime.combine(last_day + timedelta(days=1), datetime.min.time(), UTC)
    decisions = [t for t in clock.decisions if first <= t < stop]
    if not decisions or len(decisions) * len(original) > 8_000_000:
        raise ValueError("El calendario está vacío o supera el presupuesto")
    begins = _available(date(2022, 1, 13), "America/New_York", clock)
    transition = _available(date(2022, 11, 10), "America/New_York", clock)
    old_rows, schema = _source_rows(
        base, receipts[0], original, _TARGET, decisions, first, stop, transition
    )
    earlier_rows, earlier_schema = _source_rows(
        v3, receipts[1], earlier, _V3, decisions, first, stop, begins
    )
    if schema != earlier_schema:
        raise ValueError("Los paneles deben conservar el mismo esquema y precisión")
    entries = [dict(entry) for entry in original.values()]
    replacement = next(entry for entry in entries if entry["id"] == _TARGET)
    replacement.update(
        stress_history_identity(),
        name="Índice de tensión financiera de St. Louis, versión publicada",
        verified_on=datetime.now(UTC).date().isoformat(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "history"
        stage.mkdir()

        def batches():
            pending = []
            for moment in decisions:
                source = earlier_rows if moment < transition else old_rows
                pending.append(source[moment] | {"indicator_id": _TARGET})
                if len(pending) == 512:
                    yield pa.Table.from_pylist(pending, schema=schema)
                    pending = []
            if pending:
                yield pa.Table.from_pylist(pending, schema=schema)

        count = atomic_parquet_batches(stage / "macro.parquet", batches())
        with (stage / "catalog.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=entries[0])
            writer.writeheader()
            writer.writerows(entries)
            stream.flush()
            os.fsync(stream.fileno())
        report = dict(
            schema_version=1,
            policy=STRESS_HISTORY_POLICY,
            market=market,
            start=start,
            end=end,
            rows=count,
            indicators=1,
            catalog_indicators=140,
            sha256=sha256(stage / "macro.parquet"),
            catalog_sha256=sha256(stage / "catalog.csv"),
            original_catalog_sha256=identities[0]["catalog_sha256"],
            lineage=[
                dict(
                    series_id="STLFSI3",
                    source_url="https://fred.stlouisfed.org/series/STLFSI3",
                    indicator_id=_V3,
                    start=begins.isoformat(),
                    end_exclusive=transition.isoformat(),
                    **identities[1],
                ),
                dict(
                    series_id="STLFSI4",
                    source_url="https://fred.stlouisfed.org/series/STLFSI4",
                    indicator_id=_TARGET,
                    start=transition.isoformat(),
                    end_exclusive=None,
                    **identities[0],
                ),
            ],
            code_sha256={
                name: sha256(Path(__file__).with_name(name))
                for name in (
                    "macro_stress_history.py",
                    "macro_model_vintages.py",
                    "macro.py",
                    "macro_coverage.py",
                    "batches.py",
                    "storage.py",
                )
            },
            value_transform="identity",
            admission_required=True,
            population_ready=False,
            final_test_opened=False,
            limits=[
                "Los niveles de ambas versiones no se han reescalado "
                "ni acreditado como equivalentes."
            ],
        )
        for paths, identity in zip(
            ((base, base_report, base_catalog), (v3, v3_report, v3_catalog)),
            identities,
            strict=True,
        ):
            current = [sha256(path) for path in paths]
            if current != [
                identity[key] for key in ("panel_sha256", "report_sha256", "catalog_sha256")
            ]:
                raise ValueError("Una fuente ha cambiado durante la composición")
        atomic_json(stage / "report.json", report)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return report
