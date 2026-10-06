"""Factor CSI 300 retrospectivo a partir de las tablas mensuales públicas de SSE."""

import argparse
import calendar
import hashlib
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qsl, urlsplit

import exchange_calendars
import pyarrow as pa
import pyarrow.parquet as pq

from .cohort_files import _unique, read_manifest, safe_destination
from .macro_coverage import _publish_directory
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256
from .temporal import MarketClock, aware

_PAGE = "https://www.sse.com.cn/aboutus/publication/monthly/index/"
_QUERY = "COMMON_SSE_ZQZS_M_CSI300_INDEX_C"
_MAX_PAYLOAD = 1024**2
_MAX_MONTHS = 600
_SCHEMA = pa.schema(
    [("session", pa.string())]
    + [(name, pa.float64()) for name in ("open", "high", "low", "close")]
    + [("available_at", pa.timestamp("us", tz="UTC")), ("source_sha256", pa.string())]
)
_CODE = (
    "csi300_factor.py",
    "cohort_files.py",
    "macro_coverage.py",
    "preparation.py",
    "storage.py",
    "temporal.py",
)


def _stamp(value):
    if not isinstance(value, str):
        raise ValueError("La adquisición necesita una fecha UTC explícita")
    return aware(datetime.fromisoformat(value))


def _read(path, limit):
    safe_destination(path)
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("La fuente no es regular o supera el presupuesto")
    with path.open("rb") as stream:
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError("La fuente cambió y supera el presupuesto")
    return payload, hashlib.sha256(payload).hexdigest()


def _url(address, month):
    if not isinstance(address, str):
        raise ValueError("Falta la dirección de la consulta SSE")
    url = urlsplit(address)
    pairs = parse_qsl(url.query, keep_blank_values=True)
    expected = dict(sqlId=_QUERY, isPagination="false", MDATE=month, jsonCallBack="monthlyData")
    if (
        url.scheme != "https"
        or url.netloc != "query.sse.com.cn"
        or url.path != "/commonQuery.do"
        or url.fragment
        or len(pairs) != len(expected)
        or dict(pairs) != expected
    ):
        raise ValueError("La consulta no identifica el CSI 300 y el mes declarado")


def _quote(raw, month, clock, signature):
    session = raw.get("MDATE")
    if not isinstance(session, str) or not re.fullmatch(month + r"\d{2}", session):
        raise ValueError("La fecha no pertenece al mes declarado")
    day = datetime.strptime(session, "%Y%m%d").date()
    values = []
    for name in ("OPEN", "HIGH", "LOW", "CLS"):
        value = raw.get(name)
        if not isinstance(value, str) or not re.fullmatch(r"\d{1,12}\.\d{2}", value):
            raise ValueError("La cotización debe conservar los dos decimales publicados")
        values.append(float(value))
    opening, high, low, close = values
    if (
        not all(math.isfinite(v) and v > 0 for v in values)
        or not low <= min(opening, close) <= max(opening, close) <= high
    ):
        raise ValueError("Las cotizaciones OHLC no son válidas")
    return dict(
        session=day.isoformat(),
        open=opening,
        high=high,
        low=low,
        close=close,
        available_at=clock.decision(day),
        source_sha256=signature,
    )


def _month(source, entry, clock, acquired, sources):
    month = entry.get("month")
    if (
        not isinstance(month, str)
        or not re.fullmatch(r"\d{6}", month)
        or not "199001" <= month <= "202312"
    ):
        raise ValueError("El mes es inválido o cruza la reserva final")
    datetime.strptime(month, "%Y%m")
    body_path, receipt_path = source / f"{month}.jsonp", source / f"{month}.json"
    safe_destination(receipt_path)
    receipt, receipt_hash = read_manifest(receipt_path, maximum=64 * 1024)
    if not isinstance(receipt, dict) or receipt_hash != entry.get("receipt_sha256"):
        raise ValueError("La huella del recibo de descarga no coincide")
    _url(receipt.get("url"), month)
    _url(receipt.get("final_url"), month)
    payload, digest = _read(body_path, _MAX_PAYLOAD)
    if (
        type(receipt.get("status")) is not int
        or receipt["status"] != 200
        or type(receipt.get("bytes")) is not int
        or receipt["bytes"] != len(payload)
        or digest != receipt.get("sha256")
        or digest != entry.get("sha256")
        or _stamp(receipt.get("requested_at_utc")) > acquired
    ):
        raise ValueError("La descarga no conserva su contenido, estado o fecha")
    text = payload.decode("utf-8").strip()
    if not text.startswith("monthlyData(") or not text.endswith(")"):
        raise ValueError("La respuesta no conserva el envoltorio JSONP esperado")
    content = json.loads(text[len("monthlyData(") : -1], object_pairs_hook=_unique)
    if (
        not isinstance(content, dict)
        or content.get("actionErrors")
        or content.get("fieldErrors")
        or content.get("error")
    ):
        raise ValueError("La respuesta contiene un error de consulta")
    rows = content.get("result")
    if (
        not isinstance(rows, list)
        or not 1 <= len(rows) <= 31
        or type(entry.get("rows")) is not int
        or len(rows) != entry["rows"]
        or not all(isinstance(row, dict) for row in rows)
    ):
        raise ValueError("La respuesta no conserva su población diaria")
    sources[body_path], sources[receipt_path] = digest, receipt_hash
    return [_quote(row, month, clock, digest) for row in rows], len(payload)


def _inputs(acquisition, clock):
    safe_destination(acquisition)
    meta, digest = read_manifest(acquisition, maximum=_MAX_PAYLOAD)
    if (
        not isinstance(meta, dict)
        or type(meta.get("schema_version")) is not int
        or meta["schema_version"] != 1
        or meta.get("status") != "completed"
        or meta.get("source_url") != _PAGE
        or meta.get("query_id") != _QUERY
        or meta.get("training_ready") is not False
        or meta.get("final_test_opened") is not False
        or not isinstance(meta.get("months"), list)
        or not 1 <= len(meta["months"]) <= _MAX_MONTHS
    ):
        raise ValueError("La adquisición mensual no está completada o identificada")
    acquired = _stamp(meta.get("acquired_at_utc"))
    rows, sources, months, size = [], {acquisition: digest}, [], 0
    for entry in meta["months"]:
        if not isinstance(entry, dict):
            raise ValueError("El mes no tiene un recibo válido")
        quotes, byte_count = _month(acquisition.parent, entry, clock, acquired, sources)
        month = entry["month"]
        if month in months:
            raise ValueError("La adquisición contiene un mes repetido")
        months.append(month)
        rows.extend(quotes)
        size += byte_count
        if size > 64 * 1024**2:
            raise ValueError("La adquisición supera el presupuesto total de bytes")
    rows.sort(key=lambda r: r["session"])
    sessions = {r["session"] for r in rows}
    if len(sessions) != len(rows):
        raise ValueError("Hay una fecha repetida en el factor")
    if (
        type(meta.get("rows")) is not int
        or meta["rows"] != len(rows)
        or meta.get("first") != rows[0]["session"].replace("-", "")
        or meta.get("last") != rows[-1]["session"].replace("-", "")
    ):
        raise ValueError("La población y extremos no concilian con la adquisición")
    first, last = min(months), max(months)
    last_year, last_month = int(last[:4]), int(last[4:])
    last_day = calendar.monthrange(last_year, last_month)[1]
    coverage = MarketClock(
        "CN", f"{first[:4]}-{first[4:]}-01", f"{last_year:04}-{last_month:02}-{last_day:02}"
    )
    if not set(coverage.days) <= set(clock.days):
        raise ValueError("El calendario no cubre íntegramente los meses declarados")
    expected = {day.isoformat() for day in coverage.days}
    return (
        rows,
        sources,
        dict(
            months=sorted(months),
            acquisition_sha256=digest,
            acquired_at_utc=meta["acquired_at_utc"],
            source_bytes=size,
            missing_sessions=sorted(expected - sessions),
        ),
    )


def _verify(sources, code):
    for path, digest in sources.items():
        if _read(path, _MAX_PAYLOAD)[1] != digest:
            raise ValueError("Una fuente o artefacto cambió durante la materialización")
    if any(sha256(Path(__file__).with_name(name)) != digest for name, digest in code.items()):
        raise ValueError("El código cambió durante la materialización")


def materialize_csi300_factor(acquisition, output, *, clock):
    """Publicar OHLC y procedencia sin confundir la reconstrucción con vintages diarios."""
    acquisition, output = Path(acquisition), Path(output)
    if clock.market != "CN":
        raise ValueError("El factor requiere el calendario CN")
    safe_destination(output)
    for source in (acquisition.parent, Path("dataset")):
        outside_source(source, output)
        outside_source(output, source)
    code = {name: sha256(Path(__file__).with_name(name)) for name in _CODE}
    rows, sources, identity = _inputs(acquisition, clock)
    identity.update(
        code=code,
        pyarrow=pa.__version__,
        exchange_calendars=exchange_calendars.__version__,
        calendar_sha256=hashlib.sha256(
            "|".join(t.isoformat() for t in clock.decisions).encode()
        ).hexdigest(),
    )
    table = pa.Table.from_pylist(rows, schema=_SCHEMA)
    report = dict(
        schema_version=1,
        kind="retrospective_market_factor",
        market="CN",
        symbol="000300",
        source_url=_PAGE,
        identity=identity,
        rows=len(rows),
        unit="index_points",
        first_session=rows[0]["session"],
        last_session=rows[-1]["session"],
        missing_sessions=identity["missing_sessions"],
        availability_policy="retrospective_session_close_plus_5_minutes",
        adjustments="as_published_price_index_no_additional_adjustment",
        return_convention="close_over_open_minus_one",
        point_in_time_verified=False,
        financial_simulation_ready=False,
        training_ready=False,
        final_test_opened=False,
    )
    if output.exists():
        old, signature = read_manifest(output / "report.json", maximum=_MAX_PAYLOAD)
        if not isinstance(old, dict) or any(old.get(k) != v for k, v in report.items()):
            raise ValueError("La edición existente no corresponde al factor solicitado")
        artifacts = old.get("artifacts", {})
        if set(artifacts) != {"prices.parquet", "market-factors.json"}:
            raise ValueError("La edición no conserva sus artefactos")
        _verify(
            {
                **sources,
                output / "report.json": signature,
                **{output / name: digest for name, digest in artifacts.items()},
            },
            code,
        )
        with pq.ParquetFile(output / "prices.parquet") as file:
            if (
                file.schema_arrow != _SCHEMA
                or file.metadata.num_rows != len(rows)
                or file.num_row_groups > len(rows)
                or sum(
                    file.metadata.row_group(i).total_byte_size for i in range(file.num_row_groups)
                )
                > 64 * 1024**2
            ):
                raise ValueError("El Parquet existente no conserva esquema, filas o presupuesto")
            if not file.read(use_threads=False).equals(table):
                raise ValueError("El Parquet existente no coincide con las cotizaciones publicadas")
        expected_spec = _specification(output, artifacts["prices.parquet"])
        if read_manifest(output / "market-factors.json")[0] != expected_spec:
            raise ValueError("El descriptor no conserva la identidad del factor")
        _verify(
            {
                **sources,
                output / "report.json": signature,
                **{output / name: digest for name, digest in artifacts.items()},
            },
            code,
        )
        return {**old, "reused": True}
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        stage = Path(temporary) / "factor"
        stage.mkdir()
        atomic_parquet(stage / "prices.parquet", table)
        prices_hash = sha256(stage / "prices.parquet")
        atomic_json(stage / "market-factors.json", _specification(output, prices_hash))
        report["artifacts"] = {
            name: sha256(stage / name) for name in ("prices.parquet", "market-factors.json")
        }
        atomic_json(stage / "report.json", report)
        _verify(sources, code)
        _publish_directory(stage, output)
        descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {**report, "reused": False}


def _specification(output, digest):
    return {
        "CN": dict(
            market="CN",
            symbol="000300",
            unit="index_points",
            prices_path=str((output / "prices.parquet").resolve()),
            prices_sha256=digest,
            return_convention="close_over_open_minus_one",
            point_in_time_verified=False,
        )
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acquisition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calendar-start", default="1990-12-19")
    parser.add_argument("--calendar-end", default="2023-12-31")
    args = parser.parse_args(argv)
    result = materialize_csi300_factor(
        args.acquisition,
        args.output,
        clock=MarketClock("CN", args.calendar_start, args.calendar_end),
    )
    print(
        f"CSI 300: {result['rows']} sesiones. Huecos declarados: {len(result['missing_sessions'])}."
    )


if __name__ == "__main__":
    main()
