"""Ampliación del CSI 300 con tablas históricas revisadas fuera del importador."""

import calendar
import csv
import hashlib
import io
import json
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import exchange_calendars
import pyarrow as pa
import pyarrow.parquet as pq

from .cohort_files import _unique, safe_destination
from .csi300_factor import _CODE, _PAGE, _SCHEMA, _quote, _read, _specification
from .macro_coverage import _publish_directory
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock

_MAX_PDF = 32 * 1024**2
_MAX_CSV = 1024**2
_MAX_MONTHS = 600
_MAX_ROWS = 600 * 31
_UNSPECIFIED = object()
_COLUMNS = (
    "session",
    "open",
    "high",
    "low",
    "close",
    "source_pdf_sha256",
    "pdf_page_1_based",
    "printed_page",
)
_POLICY = dict(
    schema_version=1,
    kind="retrospective_market_factor",
    market="CN",
    symbol="000300",
    source_url=_PAGE,
    unit="index_points",
    availability_policy="retrospective_session_close_plus_5_minutes",
    adjustments="as_published_price_index_no_additional_adjustment",
    return_convention="close_over_open_minus_one",
    point_in_time_verified=False,
    financial_simulation_ready=False,
    training_ready=False,
    final_test_opened=False,
)


def _source(path, limit, sources, expected=_UNSPECIFIED):
    if expected is not _UNSPECIFIED and (
        not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
    ):
        raise ValueError("La fuente necesita una huella SHA256 válida")
    payload, signature = _read(path, limit)
    if expected is not _UNSPECIFIED and signature != expected:
        raise ValueError("La fuente cambió respecto de su huella declarada")
    if path in sources and sources[path][0] != signature:
        raise ValueError("Una fuente cambió entre dos lecturas")
    sources[path] = (signature, limit)
    return payload, signature


def _json(path, sources):
    payload, signature = _source(path, 1024**2, sources)
    return json.loads(payload, object_pairs_hook=_unique), signature


def _same_document(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


def _month(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"\d{6}", value)
        or not "199001" <= value <= "202312"
    ):
        raise ValueError("El mes es inválido o cruza la reserva final")
    datetime.strptime(value, "%Y%m")
    return value


def _coverage(months, clock):
    first, last = min(months), max(months)
    year, month = int(last[:4]), int(last[4:])
    end = f"{year:04}-{month:02}-{calendar.monthrange(year, month)[1]:02}"
    expected = MarketClock("CN", f"{first[:4]}-{first[4:]}-01", end)
    if not set(expected.days) <= set(clock.days):
        raise ValueError("El calendario no cubre íntegramente los meses declarados")
    return {day.isoformat() for day in expected.days}


def _relative(root, value):
    if not isinstance(value, str) or not value or len(value) > 4096 or "\\" in value:
        raise ValueError("La ruta revisada debe ser relativa y acotada")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("La ruta revisada no está confinada al manifiesto")
    path = root / relative
    safe_destination(path)
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("La ruta revisada no pertenece al manifiesto")
    return path


def _reviewed(path, clock, sources):
    meta, signature = _json(path, sources)
    if (
        not isinstance(meta, dict)
        or type(meta.get("schema_version")) is not int
        or meta["schema_version"] != 1
        or meta.get("kind") != "reviewed_csi300_history"
        or meta.get("market") != "CN"
        or meta.get("symbol") != "000300"
        or not isinstance(meta.get("sources"), list)
        or not 1 <= len(meta["sources"]) <= _MAX_MONTHS
    ):
        raise ValueError("El manifiesto no identifica una revisión acotada del CSI 300")
    rows, months = [], set()
    for entry in meta["sources"]:
        if not isinstance(entry, dict):
            raise ValueError("La fuente revisada no es un registro")
        month = _month(entry.get("month"))
        if month in months:
            raise ValueError("Hay un mes revisado repetido")
        months.add(month)
        title = entry.get("title")
        address = entry.get("source_url")
        if (
            not isinstance(title, str)
            or len(title) > 128
            or re.sub(r"\s+", "", title).casefold()
            not in {"csi300", "csi300index", "沪深300指数", "沪深300指数shse-szse300index"}
            or not isinstance(address, str)
            or len(address) > 2048
        ):
            raise ValueError("La fuente no identifica la tabla CSI 300")
        url = urlsplit(address)
        if (
            url.scheme != "https"
            or url.netloc != "www.sse.com.cn"
            or not url.path.lower().endswith(".pdf")
            or url.query
            or url.fragment
        ):
            raise ValueError("El PDF necesita una dirección primaria de SSE")
        for name in ("pdf_page_1_based", "printed_page"):
            if type(entry.get(name)) is not int or not 1 <= entry[name] <= 10000:
                raise ValueError("El localizador de página no es válido")
        for name in ("pdf_sha256", "csv_sha256"):
            if not isinstance(entry.get(name), str) or not re.fullmatch(
                r"[0-9a-f]{64}", entry[name]
            ):
                raise ValueError("La revisión necesita huellas PDF y CSV válidas")
        pdf = _relative(path.parent, entry.get("pdf_path"))
        payload, pdf_hash = _source(pdf, _MAX_PDF, sources, entry["pdf_sha256"])
        if not payload.startswith(b"%PDF-"):
            raise ValueError("La fuente revisada no tiene cabecera PDF")
        csv_path = _relative(path.parent, entry.get("csv_path"))
        payload, _ = _source(csv_path, _MAX_CSV, sources, entry["csv_sha256"])
        reader = csv.DictReader(io.StringIO(payload.decode("utf-8"), newline=""), strict=True)
        if reader.fieldnames != list(_COLUMNS):
            raise ValueError("El CSV debe conservar sus ocho columnas revisadas")
        count = 0
        for raw in reader:
            count += 1
            if count > 31 or set(raw) != set(_COLUMNS) or any(v is None for v in raw.values()):
                raise ValueError("La tabla mensual contiene campos o filas fuera del límite")
            if (
                raw["source_pdf_sha256"] != pdf_hash
                or raw["pdf_page_1_based"] != str(entry["pdf_page_1_based"])
                or raw["printed_page"] != str(entry["printed_page"])
                or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw["session"])
            ):
                raise ValueError("Una fila no conserva su fecha o localizador de origen")
            quote = dict(
                zip(
                    ("OPEN", "HIGH", "LOW", "CLS"),
                    (raw[n] for n in ("open", "high", "low", "close")),
                    strict=True,
                )
            )
            quote["MDATE"] = raw["session"].replace("-", "")
            rows.append(_quote(quote, month, clock, pdf_hash))
        if not count:
            raise ValueError("La tabla mensual revisada está vacía")
    if len({row["session"] for row in rows}) != len(rows):
        raise ValueError("Hay una sesión revisada repetida")
    return rows, months, dict(reviewed_manifest_sha256=signature, reviewed_sources=meta["sources"])


def _table(payload):
    with pq.ParquetFile(pa.BufferReader(payload)) as file:
        if (
            file.schema_arrow != _SCHEMA
            or not 1 <= file.metadata.num_rows <= _MAX_ROWS
            or file.num_row_groups > file.metadata.num_rows
            or sum(file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups))
            > 64 * 1024**2
        ):
            raise ValueError("El Parquet no conserva el esquema o el presupuesto del factor")
        return file.read(use_threads=False)


def _base(path, clock, sources):
    report, signature = _json(path / "report.json", sources)
    if (
        not isinstance(report, dict)
        or type(report.get("schema_version")) is not int
        or any(type(report.get(k)) is not type(v) or report[k] != v for k, v in _POLICY.items())
        or not isinstance(report.get("identity"), dict)
        or type(report.get("rows")) is not int
        or not 1 <= report["rows"] <= _MAX_ROWS
        or not isinstance(report.get("artifacts"), dict)
        or set(report["artifacts"]) != {"prices.parquet", "market-factors.json"}
    ):
        raise ValueError("La edición base no conserva el contrato retrospectivo CSI 300")
    months = report["identity"].get("months")
    if not isinstance(months, list) or not 1 <= len(months) <= _MAX_MONTHS:
        raise ValueError("La edición base no declara sus meses")
    months = [_month(month) for month in months]
    if len(set(months)) != len(months):
        raise ValueError("La edición base repite un mes")
    expected = _coverage(months, clock)
    if report.get("first_session") not in expected or report.get("last_session") not in expected:
        raise ValueError("La edición base declara extremos ajenos al calendario")
    prices_hash = report["artifacts"]["prices.parquet"]
    payload, _ = _source(path / "prices.parquet", 32 * 1024**2, sources, prices_hash)
    table = _table(payload)
    payload, _ = _source(
        path / "market-factors.json", 1024**2, sources, report["artifacts"]["market-factors.json"]
    )
    if not _same_document(
        json.loads(payload, object_pairs_hook=_unique), _specification(path, prices_hash)
    ):
        raise ValueError("El descriptor base no corresponde a sus cotizaciones")
    rows = table.to_pylist()
    sessions = [row["session"] for row in rows]
    if (
        not all(isinstance(day, str) and day in expected for day in sessions)
        or sessions != sorted(set(sessions))
        or len(rows) != report["rows"]
        or sessions[0] != report["first_session"]
        or sessions[-1] != report["last_session"]
        or sorted(expected - set(sessions)) != report.get("missing_sessions")
    ):
        raise ValueError("La edición base no concilia fechas, población o huecos")
    for row in rows:
        digest = row["source_sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("La fila base no conserva una huella de procedencia")
        values = [row[name] for name in ("open", "high", "low", "close")]
        if any(value is None for value in values):
            raise ValueError("Falta una cotización base")
        raw = dict(
            zip(("OPEN", "HIGH", "LOW", "CLS"), (f"{value:.2f}" for value in values), strict=True)
        )
        raw["MDATE"] = row["session"].replace("-", "")
        if _quote(raw, raw["MDATE"][:6], clock, digest) != row:
            raise ValueError("La fila base cambia precisión, OHLC o disponibilidad")
    return rows, set(months), signature


def _verify(sources):
    for path, (signature, limit) in sources.items():
        if _read(path, limit)[1] != signature:
            raise ValueError("Una fuente o artefacto cambió durante la ampliación")


def extend_csi300_history(base_edition, reviewed_manifest, output, *, clock):
    """Unir tablas revisadas y una edición base sin reemplazar fuentes o fechas maduras."""
    base, manifest, output = map(Path, (base_edition, reviewed_manifest, output))
    if clock.market != "CN":
        raise ValueError("La ampliación requiere el calendario CN")
    for path in (base, manifest, output):
        safe_destination(path)
    for source in (base, manifest.parent, Path("dataset")):
        outside_source(source, output)
        outside_source(output, source)
    sources = {}
    code = {}
    for name in (*_CODE, "csi300_history.py"):
        _, code[name] = _source(Path(__file__).with_name(name), 1024**2, sources)
    rows, months, base_hash = _base(base, clock, sources)
    try:
        reviewed, new_months, identity = _reviewed(manifest, clock, sources)
    except csv.Error as error:
        raise ValueError("El CSV revisado no tiene una estructura válida") from error
    expected = _coverage(months | new_months, clock)
    merged = {row["session"]: row for row in rows}
    overlaps = 0
    for row in reviewed:
        prior = merged.get(row["session"])
        if prior is not None:
            if any(prior[name] != row[name] for name in ("open", "high", "low", "close")):
                raise ValueError("Un solapamiento contradice las cotizaciones de la base")
            overlaps += 1
        else:
            merged[row["session"]] = row
    ordered = [merged[session] for session in sorted(merged)]
    identity.update(
        base_report_sha256=base_hash,
        months=sorted(months | new_months),
        code=code,
        pyarrow=pa.__version__,
        exchange_calendars=exchange_calendars.__version__,
        calendar_sha256=hashlib.sha256(
            "|".join(t.isoformat() for t in clock.decisions).encode()
        ).hexdigest(),
    )
    table = pa.Table.from_pylist(ordered, schema=_SCHEMA)
    report = dict(
        **_POLICY,
        identity=identity,
        rows=len(ordered),
        base_rows=len(rows),
        reviewed_rows=len(reviewed),
        identical_overlap_rows=overlaps,
        added_rows=len(ordered) - len(rows),
        first_session=ordered[0]["session"],
        last_session=ordered[-1]["session"],
        missing_sessions=sorted(expected - merged.keys()),
    )
    if output.exists():
        old, _ = _json(output / "report.json", sources)
        if (
            not isinstance(old, dict)
            or set(old) != set(report) | {"artifacts"}
            or not _same_document({key: old[key] for key in report}, report)
            or not isinstance(old["artifacts"], dict)
            or set(old["artifacts"]) != {"prices.parquet", "market-factors.json"}
        ):
            raise ValueError("La salida existente no corresponde a esta ampliación")
        artifacts = old["artifacts"]
        payload, _ = _source(
            output / "prices.parquet", 32 * 1024**2, sources, artifacts["prices.parquet"]
        )
        if not _table(payload).equals(table):
            raise ValueError("El contenido existente no coincide con las cotizaciones revisadas")
        payload, _ = _source(
            output / "market-factors.json", 1024**2, sources, artifacts["market-factors.json"]
        )
        if not _same_document(
            json.loads(payload, object_pairs_hook=_unique),
            _specification(output, artifacts["prices.parquet"]),
        ):
            raise ValueError("El descriptor existente no corresponde a esta ampliación")
        _verify(sources)
        return {**old, "reused": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "factor"
        stage.mkdir()
        atomic_parquet(stage / "prices.parquet", table)
        atomic_json(
            stage / "market-factors.json", _specification(output, sha256(stage / "prices.parquet"))
        )
        report["artifacts"] = {
            name: sha256(stage / name) for name in ("prices.parquet", "market-factors.json")
        }
        atomic_json(stage / "report.json", report)
        _verify(sources)
        safe_destination(output)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {**report, "reused": False}
