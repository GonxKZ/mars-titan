"""Validación acotada de formatos, sin transformar observaciones para modelos."""

from __future__ import annotations

import csv
import io
import math
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile
import zlib
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from pathlib import PurePosixPath

MAX_UNCOMPRESSED_BYTES = 50 * 1024 * 1024
MAX_ZIP_MEMBERS = 1000
SUPPORTED_VALIDATORS = {
    "fred_csv",
    "ecb_csv",
    "vix_csv",
    "gscpi_csv",
    "stooq_csv",
    "fed_rss",
    "treasury_xml",
    "french_zip",
    "gscpi_xlsx",
    "financial_pdf",
}
MISSING = {"", ".", "NA", "N/A", "NAN", "#N/A"}


def reference_date(value: str) -> str | None:
    for pattern in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d", "%Y-%m", "%d-%b-%Y"):
        try:
            return datetime.strptime(value.strip(), pattern).date().isoformat()
        except ValueError:
            continue
    return None


def date_stats(values: list[str], as_of: date) -> dict:
    return {
        "date_min": min(values),
        "date_max": max(values),
        "dated_records": len(values),
        "duplicate_dates": len(values) - len(set(values)),
        "reference_dates_after_snapshot_date": sum(d > as_of.isoformat() for d in values),
    }


def numeric(value: str) -> bool:
    """Devuelve si falta el valor y rechaza números no finitos no declarados ausentes."""
    value = value.strip()
    if value.upper() in MISSING:
        return True
    if not math.isfinite(float(value)):
        raise ValueError("Valor numérico infinito o no finito")
    return False


def csv_rows(content: bytes) -> list[list[str]]:
    text = content.decode("utf-8-sig")
    if re.search(r"<(?:!doctype|html|head|body)\b", text[:4096], flags=re.I):
        raise ValueError("HTML recibido donde se esperaba CSV")
    rows = list(csv.reader(io.StringIO(text), strict=True))
    if not rows:
        raise ValueError("CSV vacío")
    return rows


def numeric_csv(content: bytes, kind: str, as_of: date) -> dict:
    rows = csv_rows(content)
    header = [value.strip() for value in rows[0]]
    if len(header) < 2 or len(set(header)) != len(header) or not all(header):
        raise ValueError("Cabecera CSV vacía, duplicada o insuficiente")
    date_position = 0
    positions = list(range(1, len(header)))
    if kind == "ecb_csv":
        if not {"TIME_PERIOD", "OBS_VALUE"} <= set(header):
            raise ValueError("CSV BCE sin TIME_PERIOD y OBS_VALUE")
        date_position = header.index("TIME_PERIOD")
        positions = [header.index("OBS_VALUE")]
    elif kind == "fred_csv" and header[0].lower() != "sasdate":
        raise ValueError("Cabecera FRED sin sasdate")
    elif kind == "vix_csv" and header != ["DATE", "OPEN", "HIGH", "LOW", "CLOSE"]:
        raise ValueError("Cabecera VIX inesperada")
    elif kind == "stooq_csv" and header != ["Date", "Open", "High", "Low", "Close", "Volume"]:
        raise ValueError("Cabecera Stooq OHLCV inesperada")
    elif kind == "gscpi_csv":
        if header[0].lower() != "date":
            raise ValueError("Cabecera GSCPI sin Date")
        for label in header[1:]:
            datetime.strptime(label, "%b-%y")
    dates, missing, metadata_rows = [], 0, 0
    for row in rows[1:]:
        if not row or not any(value.strip() for value in row):
            metadata_rows += 1
            continue
        if len(row) != len(header):
            raise ValueError("Filas CSV con ancho irregular")
        if kind == "fred_csv" and row[0].strip().lower().rstrip(":") in {
            "transform",
            "factors",
        }:
            metadata_rows += 1
            continue
        observed = reference_date(row[date_position])
        if observed is None:
            raise ValueError("Fecha CSV no reconocible")
        dates.append(observed)
        missing += sum(numeric(row[position]) for position in positions)
    if not dates:
        raise ValueError("CSV sin observaciones fechadas")
    result = {
        "valid": True,
        "validator": kind,
        "row_count": len(dates),
        "column_count": len(header),
        "columns_sample": header[:12],
        "numeric_cells_checked": len(dates) * len(positions),
        "missing_numeric_cells": missing,
        "metadata_or_blank_rows": metadata_rows,
        "date_semantics": "reference_period_not_publication_time",
        **date_stats(dates, as_of),
    }
    if kind == "gscpi_csv":
        result.update(vintage_labels=header[1:], vintage_timestamp_precision="month_label_only")
    return result


def safe_xml(content: bytes) -> ET.Element:
    text = content.decode("utf-8-sig")
    if "\x00" in text or re.search(r"<!\s*(?:DOCTYPE|ENTITY)\b", text, flags=re.I):
        raise ValueError("XML con DOCTYPE/ENTITY o codificación no admitida")
    return ET.fromstring(text)


def bounded_zip(content: bytes) -> zipfile.ZipFile:
    zipped = zipfile.ZipFile(io.BytesIO(content))
    try:
        members = zipped.infolist()
        if len(members) > MAX_ZIP_MEMBERS or len({m.filename for m in members}) != len(members):
            raise ValueError("ZIP con exceso de miembros o nombres duplicados")
        if sum(m.file_size for m in members) > MAX_UNCOMPRESSED_BYTES:
            raise ValueError("ZIP supera el límite de bytes descomprimidos")
        for member in members:
            if member.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValueError("Método de compresión ZIP no admitido")
            path = PurePosixPath(member.filename)
            if path.is_absolute() or ".." in path.parts or member.flag_bits & 1:
                raise ValueError("ZIP con rutas no permitidas o cifrado")
            if member.file_size > 1024 * 1024 and member.file_size > 200 * max(
                1, member.compress_size
            ):
                raise ValueError("ZIP con relación de descompresión excesiva")
        if zipped.testzip() is not None:
            raise ValueError("ZIP con error CRC")
        return zipped
    except Exception:
        zipped.close()
        raise


def french_csv(content: bytes, as_of: date) -> dict:
    with bounded_zip(content) as zipped:
        members = [n for n in zipped.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("ZIP de factores de Fama y French sin un CSV único")
        rows = csv_rows(zipped.read(members[0]))
    header = ["", "Mkt-RF", "SMB", "HML", "RF"]
    positions = [i for i, row in enumerate(rows) if [v.strip() for v in row] == header]
    if len(positions) != 1:
        raise ValueError("Factores de Fama y French sin cabecera diaria reconocible")
    dates, missing = [], 0
    for row in rows[positions[0] + 1 :]:
        if not row or not row[0].strip().isdigit():
            continue
        if len(row) != 5 or len(row[0].strip()) != 8:
            raise ValueError("Factores de Fama y French con esquema diario inesperado")
        observed = reference_date(row[0])
        if observed is None:
            raise ValueError("Fecha de los factores de Fama y French no reconocible")
        dates.append(observed)
        for value in row[1:]:
            missing += numeric(value) or float(value) in {-99.99, -999.0}
    if not dates:
        raise ValueError("Factores de Fama y French sin observaciones")
    return {
        "valid": True,
        "validator": "french_zip",
        "zip_crc": "ok",
        "member": members[0],
        "row_count": len(dates),
        "column_count": 5,
        "missing_sentinel_cells": missing,
        "unit": "percent_daily_return",
        **date_stats(dates, as_of),
        "date_semantics": "return_reference_date_not_file_publication_time",
    }


def spreadsheet(content: bytes) -> dict:
    ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with bounded_zip(content) as zipped:
        names = zipped.namelist()
        if not {"[Content_Types].xml", "xl/workbook.xml"} <= set(names):
            raise ValueError("ZIP no es un XLSX con workbook")
        trees = {name: safe_xml(zipped.read(name)) for name in names if name.endswith(".xml")}
        sheets = [s.attrib["name"] for s in trees["xl/workbook.xml"].iter(ns + "sheet")]
        counts = {}
        for name, tree in trees.items():
            if name.startswith("xl/worksheets/"):
                counts[name] = sum(bool(list(row)) for row in tree.iter(ns + "row"))
                for cell in tree.iter(ns + "c"):
                    if cell.attrib.get("t", "n") == "n" and cell.findtext(ns + "v") is not None:
                        numeric(cell.findtext(ns + "v"))
    if not sheets or not any(counts.values()):
        raise ValueError("XLSX sin hojas y filas con celdas")
    return {
        "valid": True,
        "validator": "gscpi_xlsx",
        "sheet_names": sheets,
        "worksheet_nonempty_row_counts": counts,
        "zip_crc": "ok",
        "scope": (
            "Se validan el contenedor y las celdas, sin evaluar fórmulas ni interpretar "
            "fechas de Excel."
        ),
    }


def xml_data(content: bytes, kind: str, as_of: date) -> dict:
    root = safe_xml(content)
    if kind == "fed_rss":
        items = root.findall("./channel/item")
        if not items or not all(i.findtext("title") and i.findtext("link") for i in items):
            raise ValueError("RSS sin titulares y enlaces")
        dates = []
        for item in items:
            stamp = parsedate_to_datetime(item.findtext("pubDate", ""))
            if stamp.tzinfo is None:
                raise ValueError("pubDate sin zona horaria")
            dates.append(stamp.astimezone(UTC).isoformat())
        return {
            "valid": True,
            "validator": kind,
            "row_count": len(items),
            "published_at_min_utc": min(dates),
            "published_at_max_utc": max(dates),
            "publisher_full_text_downloaded": False,
        }
    ns = {
        "a": "http://www.w3.org/2005/Atom",
        "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
        "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
    }
    dates, numeric_count = [], 0
    for entry in root.findall("a:entry", ns):
        props = entry.find("a:content/m:properties", ns)
        if props is None:
            raise ValueError("Entrada Treasury sin properties")
        observed = reference_date(props.findtext("d:NEW_DATE", "", ns)[:10])
        if observed is None:
            raise ValueError("Treasury sin fecha reconocible")
        dates.append(observed)
        for field in props:
            if field.tag.rsplit("}", 1)[-1].startswith("BC_"):
                numeric(field.text or "")
                numeric_count += 1
    if not dates or not numeric_count:
        raise ValueError("XML Treasury sin fechas y rendimientos")
    return {
        "valid": True,
        "validator": kind,
        "row_count": len(dates),
        "numeric_cells_checked": numeric_count,
        "date_semantics": "yield_reference_date",
        **date_stats(dates, as_of),
    }


def financial_pdf(content: bytes) -> dict:
    if not content.startswith(b"%PDF-") or b"%%EOF" not in content[-4096:]:
        raise ValueError("PDF ausente o incompleto")
    info = subprocess.run(
        ["pdfinfo", "-"], input=content, capture_output=True, timeout=15, check=False
    )
    if info.returncode:
        raise ValueError("pdfinfo rechazó el PDF")
    metadata = {}
    for line in info.stdout.decode("utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key in {"Pages", "Page size", "PDF version", "Encrypted", "Title"}:
            metadata[key] = value.strip()
    pages = int(metadata.get("Pages", "0"))
    if pages < 1:
        raise ValueError("PDF sin páginas")
    return {
        "valid": True,
        "validator": "financial_pdf",
        "page_count": pages,
        "metadata": metadata,
        "table_extraction_performed": False,
        "visual_review": "pending",
    }


def validate(content: bytes, kind: str, *, as_of: date | None = None) -> dict:
    as_of = as_of or datetime.now(UTC).date()
    try:
        if kind in {"fred_csv", "ecb_csv", "vix_csv", "gscpi_csv", "stooq_csv"}:
            return numeric_csv(content, kind, as_of)
        if kind in {"fed_rss", "treasury_xml"}:
            return xml_data(content, kind, as_of)
        if kind == "french_zip":
            return french_csv(content, as_of)
        if kind == "gscpi_xlsx":
            return spreadsheet(content)
        if kind == "financial_pdf":
            return financial_pdf(content)
    except (
        UnicodeError,
        csv.Error,
        ET.ParseError,
        zipfile.BadZipFile,
        zlib.error,
        EOFError,
        KeyError,
    ) as exc:
        raise ValueError(f"Formato {kind} inválido: {exc}") from exc
    raise ValueError(f"Validador no admitido: {kind}")
