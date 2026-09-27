"""Versiones GSCPI con disponibilidad conservadora al terminar el mes de versión."""

import argparse
import calendar
import csv
import hashlib
import io
import json
import math
import os
import re
import resource
import subprocess
import tempfile
import time
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pyarrow as pa

from .macro import calculate_macro
from .macro_coverage import _publish_directory
from .macro_model_vintages import MONTH_ABBREVIATIONS, model_vintage_contract
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

SOURCE_URL = (
    "https://www.newyorkfed.org/medialibrary/research/interactives/data/"
    "gscpi/gscpi_interactive_data.csv"
)
_MONTHS = MONTH_ABBREVIATIONS
_FIRST = date(2022, 5, 1)
_UNIT = "standard_deviations_of_provider_historical_mean"


def _month(label):
    if not re.fullmatch(r"[A-Z][a-z]{2}-\d{2}", label) or label[:3] not in _MONTHS:
        raise ValueError("La versión GSCPI debe indicar un mes y un año reconocibles")
    return date(2000 + int(label[-2:]), _MONTHS.index(label[:3]) + 1, 1)


def _month_end(moment):
    return date(moment.year, moment.month, calendar.monthrange(moment.year, moment.month)[1])


def parse_gscpi_vintages(content: bytes, *, last_vintage: str) -> list[dict]:
    """Leer columnas de versión, sin sustituirlas por la estimación revisada actual.

    La etiqueta mensual no acredita un día exacto de publicación. El límite de
    disponibilidad se calcula al final del mes y el motor aplica después su
    siguiente sesión conservadora. No se admiten las versiones preliminares
    anteriores al comienzo de la publicación mensual regular en mayo de 2022.
    """
    if not isinstance(content, bytes) or not 0 < len(content) <= 2 * 1024**2:
        raise ValueError("El CSV GSCPI está vacío o supera 2 MiB")
    if not re.fullmatch(r"20\d{2}-(0[1-9]|1[0-2])", last_vintage):
        raise ValueError("La última versión debe expresarse como YYYY-MM")
    end = date.fromisoformat(last_vintage + "-01")
    if end < _FIRST:
        raise ValueError("El periodo solicitado precede al archivo mensual admitido")
    reader = csv.reader(io.StringIO(content.decode("utf-8-sig")))
    header = next(reader, [])
    if not header or header[0] != "Date" or not 2 <= len(header) <= 1025:
        raise ValueError("La cabecera del CSV GSCPI no cumple su contrato")
    months = [_month(label) for label in header[1:]]
    if months != sorted(set(months)):
        raise ValueError("Las versiones GSCPI están duplicadas o desordenadas")
    selected = [(i + 1, month) for i, month in enumerate(months) if _FIRST <= month <= end]
    if not selected or selected[0][1] != _FIRST or selected[-1][1] != end:
        raise ValueError("El archivo no contiene el intervalo completo de versiones solicitado")
    if any(
        (b.year * 12 + b.month) - (a.year * 12 + a.month) != 1
        for (_, a), (_, b) in zip(selected[:-1], selected[1:], strict=True)
    ):
        raise ValueError("Faltan meses intermedios del archivo GSCPI")
    digest = hashlib.sha256(content).hexdigest()
    result = []
    previous = None
    for row_number, cells in enumerate(reader, start=2):
        if row_number > 2402:
            raise ValueError("El CSV GSCPI supera el presupuesto de filas")
        if not any(cell.strip() for cell in cells):
            continue
        if len(cells) != len(header):
            raise ValueError("Una fila GSCPI tiene anchura incorrecta o supera el presupuesto")
        label = cells[0]
        if not re.fullmatch(r"\d{2}-[A-Z][a-z]{2}-\d{4}", label) or label[3:6] not in _MONTHS:
            raise ValueError("La fecha mensual de GSCPI no tiene el formato esperado")
        period = date(int(label[-4:]), _MONTHS.index(label[3:6]) + 1, int(label[:2]))
        if period != _month_end(period) or previous is not None and period <= previous:
            raise ValueError("Las observaciones deben ser fines de mes únicos y ordenados")
        previous = period
        for position, (column, month) in enumerate(selected):
            raw = cells[column].strip()
            try:
                value = None if raw in {"", "#N/A"} else float(raw)
            except ValueError as error:
                raise ValueError(
                    f"Valor inválido en fila {row_number}, versión {header[column]}"
                ) from error
            if value is not None and not math.isfinite(value):
                raise ValueError("El valor GSCPI no es finito")
            if period >= month:
                if value is not None:
                    raise ValueError("Una versión GSCPI contiene un periodo futuro")
                continue
            start = _month_end(month)
            stop = (
                _month_end(selected[position + 1][1]) - timedelta(days=1)
                if position + 1 < len(selected)
                else date.max
            )
            result.append(
                dict(
                    indicator_id="global_supply_pressure",
                    period_start=period.isoformat(),
                    realtime_start=start.isoformat(),
                    realtime_end=stop.isoformat(),
                    value=value,
                    source_hash=digest,
                    source_timezone="America/New_York",
                    native_unit=_UNIT,
                    seasonal_adjustment="published_composite_index",
                    missing_reason="missing_source_value" if value is None else None,
                    vintage_label=header[column],
                    availability_precision="month",
                    availability_policy="end_of_vintage_month_then_next_session",
                    publication_timestamp_verified=False,
                )
            )
    if not result:
        raise ValueError("El CSV GSCPI no contiene observaciones de las versiones solicitadas")
    return result


def _download():
    """Descargar únicamente el CSV oficial, sin redirecciones ni respuestas ilimitadas."""
    with tempfile.TemporaryDirectory(prefix="mars-gscpi-") as directory:
        destination = Path(directory) / "source.csv"
        command = [
            "curl",
            "--silent",
            "--show-error",
            "--fail",
            "--proto",
            "=https",
            "--connect-timeout",
            "15",
            "--max-time",
            "45",
            "--max-filesize",
            str(2 * 1024**2),
            "--output",
            str(destination),
            "--write-out",
            "%{http_code}\n%{url_effective}",
            SOURCE_URL,
        ]
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=50)
        if result.returncode or result.stdout.splitlines() != ["200", SOURCE_URL]:
            raise OSError("No se pudo descargar el CSV oficial GSCPI: " + result.stderr[:500])
        if not destination.is_file() or destination.stat().st_size > 2 * 1024**2:
            raise ValueError("La descarga GSCPI supera el presupuesto de lectura")
        return destination.read_bytes()


def prepare_gscpi(output: Path, catalog_path: Path, *, market: str, start: str, end: str) -> dict:
    """Guardar fuente, política temporal y panel sin sustituir una edición existente."""
    began = time.perf_counter()
    output, catalog_path = Path(output), Path(catalog_path).resolve(strict=True)
    outside_source(Path("dataset"), output)
    outside_source(catalog_path.parent, output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("La edición GSCPI ya existe")
    first, last = date.fromisoformat(start), date.fromisoformat(end)
    if first > last or (last - first).days > 366 * 50:
        raise ValueError("El periodo GSCPI está invertido o supera cincuenta años")
    if catalog_path.stat().st_size > 2 * 1024**2:
        raise ValueError("El catálogo supera el presupuesto")
    catalog_hash = sha256(catalog_path)
    with catalog_path.open(encoding="utf-8", newline="") as stream:
        entries = [
            row
            for row in csv.DictReader(stream)
            if row["id"] in {"global_supply_pressure", "global_supply_pressure_change_1m"}
        ]
    if {row["id"] for row in entries} != {
        "global_supply_pressure",
        "global_supply_pressure_change_1m",
    } or len(entries) != 2:
        raise ValueError("El catálogo debe declarar GSCPI y su cambio mensual una sola vez")
    raw_entries = [entry for entry in entries if entry["kind"] == "raw"]
    if len(raw_entries) != 1 or raw_entries[0]["id"] != "global_supply_pressure":
        raise ValueError("El catálogo debe conservar GSCPI como serie publicada")
    raw_entry = raw_entries[0]
    _, exclusion = model_vintage_contract(raw_entry)
    if exclusion is not None:
        raise ValueError("El catálogo GSCPI no acredita la política mensual requerida")
    clock = MarketClock(market, start, end)
    if not clock.decisions:
        raise ValueError("El periodo GSCPI no contiene sesiones de mercado")
    content = _download()
    retrieved_at = datetime.now(UTC).isoformat()
    # Una versión de este mes no se admite dentro del mismo mes, ni siquiera en su último cierre.
    last_month = last.replace(day=1) - timedelta(days=1)
    vintage_end = f"{last_month.year:04d}-{last_month.month:02d}"
    records = (
        parse_gscpi_vintages(content, last_vintage=vintage_end) if last_month >= _FIRST else []
    )
    rows = calculate_macro(records, entries, clock)
    if sha256(catalog_path) != catalog_hash:
        raise ValueError("El catálogo cambió durante la preparación")
    report = dict(
        schema_version=1,
        market=market,
        start=start,
        end=end,
        source_url=SOURCE_URL,
        retrieved_at_utc=retrieved_at,
        source_sha256=hashlib.sha256(content).hexdigest(),
        catalog_sha256=catalog_hash,
        vintage_rows=len(records),
        rows=len(rows),
        decisions=len(clock.decisions),
        last_admitted_vintage=vintage_end,
        nonempty_indicators=sorted({r["indicator_id"] for r in rows if r["value"] is not None}),
        computed_values=sum(r["value"] is not None for r in rows),
        availability_policy="end_of_vintage_month_then_next_session",
        publication_timestamp_verified=False,
        source_precision="as_published_in_monthly_csv",
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "edition"
        stage.mkdir()
        with (stage / "source.csv").open("wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_parquet(stage / "macro.parquet", pa.Table.from_pylist(rows))
        report.update(
            sha256=sha256(stage / "macro.parquet"), elapsed_seconds=time.perf_counter() - began
        )
        atomic_json(stage / "report.json", report)
        _publish_directory(stage, output)
        directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--market", choices=("US", "CN"), required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    args = parser.parse_args(argv)
    result = prepare_gscpi(
        args.output, args.catalog, market=args.market, start=args.start, end=args.end
    )
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
