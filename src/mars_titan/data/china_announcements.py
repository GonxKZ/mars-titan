"""Catálogo recuperable de anuncios CNINFO, sin descargar ni admitir balances."""

import argparse
import fcntl
import hashlib
import html
import json
import os
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow as pa

from .batches import read_bounded_table
from .cohort_files import _unique, read_manifest, safe_destination
from .macro_coverage import _publish_directory
from .preparation import atomic_parquet
from .storage import atomic_json, outside_source, sha256

_URL = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
_PAGE_SIZE = 30
_MAX_PAGES = 100
_MAX_RESPONSE = 1024**2
_MAX_RECORDS = 100_000
_MAX_RESPONSES = 10_000
_MAX_BYTES = 64 * 1024**2
_CUTOFF = date(2023, 12, 31)
_CODE = (
    "china_announcements.py",
    "cohort_files.py",
    "storage.py",
    "batches.py",
    "macro_coverage.py",
    "preparation.py",
)
_SCHEMA = pa.schema(
    [
        (name, pa.string())
        for name in (
            "announcement_id",
            "sec_code",
            "symbol",
            "sec_name",
            "org_id",
            "page_column",
            "publication_date",
            "title",
            "pdf_url",
            "source_response_sha256",
        )
    ]
    + [
        ("announcement_time_ms", pa.int64()),
        ("in_census", pa.bool_()),
        ("queue_periods", pa.list_(pa.string())),
        ("source_row", pa.int32()),
    ]
)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _json(path, maximum=8 * 1024**2):
    safe_destination(path)
    return read_manifest(path, maximum=maximum)


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Falta una huella SHA256 válida")
    return value


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        raise ValueError("La fecha debe tener formato ISO de calendario")
    result = date.fromisoformat(value)
    if result > _CUTOFF:
        raise ValueError("La fecha cruza la reserva final")
    return result


def _queue(path):
    safe_destination(path)
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise ValueError("La cola no es regular o supera el presupuesto")
    signature = sha256(path)
    table = read_bounded_table(path, max_rows=_MAX_RECORDS, max_bytes=_MAX_BYTES)
    names = ("symbol", "market", "period_end")
    if not len(table) or any(
        name not in table.column_names
        or not pa.types.is_string(table[name].type)
        or table[name].null_count
        for name in names
    ):
        raise ValueError("La cola necesita símbolos, mercado y cierres explícitos")
    census, pairs = {}, set()
    for row in table.select(names).to_pylist():
        symbol, period = row["symbol"], row["period_end"]
        if row["market"] != "CN" or not re.fullmatch(r"[0-9]{6}\.(?:SZ|SS)", symbol):
            raise ValueError("La cola contiene un instrumento ajeno al censo CN")
        _day(period)
        if (symbol, period) in pairs:
            raise ValueError("La cola repite un cierre del mismo instrumento")
        pairs.add((symbol, period))
        census.setdefault(symbol, []).append(period)
    if len(census) > 1024 or sha256(path) != signature:
        raise ValueError("La cola cambió o supera el presupuesto de instrumentos")
    return {s: sorted(p) for s, p in sorted(census.items())}, signature, len(table)


def _task(start, end):
    return dict(start=start, end=end, page=1, total=None, collected=0)


def _key(task):
    return f"{task['start']}_{task['end']}_{task['page']:03}"


def _parameters(task, issuer=None):
    return dict(
        pageNum=str(task["page"]),
        pageSize=str(_PAGE_SIZE),
        column="szse",
        tabName="fulltext",
        plate="",
        stock=f"{issuer['symbol'].split('.')[0]},{issuer['org_id']}" if issuer else "",
        searchkey="",
        secid="",
        category="category_ndbg_szsh",
        trade="",
        seDate=f"{task['start']}~{task['end']}",
        sortName="",
        sortType="",
        isHLtitle="true",
    )


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _request(parameters):
    request = urllib.request.Request(
        _URL,
        data=urllib.parse.urlencode(parameters).encode(),
        method="POST",
        headers={
            "User-Agent": "MARS-TITAN research (+https://github.com/GonxKZ/mars-titan)",
            "Referer": "https://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
            "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
        },
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
    try:
        response = opener.open(request, timeout=30)
    except urllib.error.HTTPError as error:
        response = error
    with response:
        return response.status, dict(response.headers), response.read(_MAX_RESPONSE)


def _sync(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _truncated(headers, read_bytes, stored_bytes):
    if (
        not isinstance(headers, dict)
        or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items())
        or type(read_bytes) is not int
        or type(stored_bytes) is not int
        or not 0 <= stored_bytes <= _MAX_RESPONSE
        or read_bytes < stored_bytes
    ):
        raise ValueError("La respuesta no conserva cabeceras y un estado de lectura válidos")
    lengths = [v.strip(" \t") for k, v in headers.items() if k.lower() == "content-length"]
    if lengths and (
        any(not re.fullmatch(r"[0-9]+", value) for value in lengths) or len(set(lengths)) != 1
    ):
        return True
    declared = int(lengths[0]) if lengths else None
    return (
        read_bytes != stored_bytes
        or (declared is not None and declared != read_bytes)
        or (stored_bytes == _MAX_RESPONSE and declared != _MAX_RESPONSE)
    )


def _save_response(directory, task, configuration, response, requested_at, injected, issuer=None):
    status, headers, body = response
    if type(status) is not int or not 100 <= status <= 599 or not isinstance(body, bytes):
        raise ValueError("El transporte no devolvió un estado HTTP y bytes válidos")
    if not isinstance(headers, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in headers.items()
    ):
        raise ValueError("El transporte no devolvió cabeceras textuales")
    kept = {
        k.lower(): v
        for k, v in headers.items()
        if k.lower()
        in {
            "content-type",
            "content-length",
            "date",
            "retry-after",
            "last-modified",
            "etag",
        }
    }
    if any(len(value) > 4096 for value in kept.values()):
        raise ValueError("Las cabeceras HTTP superan su presupuesto")
    read_bytes = len(body)
    body = body[:_MAX_RESPONSE]
    truncated = _truncated(kept, read_bytes, len(body))
    receipt = dict(
        schema_version=1,
        configuration_sha256=configuration,
        url=_URL,
        method="POST",
        parameters=_parameters(task, issuer),
        status=status,
        headers=kept,
        requested_at_utc=requested_at,
        bytes=len(body),
        read_bytes=read_bytes,
        sha256=hashlib.sha256(body).hexdigest(),
        truncated=truncated,
        transport="injected" if injected else "urllib_public",
    )
    safe_destination(directory)
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".response-", dir=directory.parent) as temporary:
        stage = Path(temporary) / "response"
        stage.mkdir()
        with (stage / "body.json").open("xb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        atomic_json(stage / "receipt.json", receipt)
        _publish_directory(stage, directory)
        _sync(directory.parent)


def _response(directory, task, configuration, sources, expected=None, issuer=None):
    receipt, signature = _json(directory / "receipt.json", 64 * 1024)
    if expected is not None and signature != _hash(expected):
        raise ValueError("Cambió el recibo de una respuesta confirmada")
    if (
        not isinstance(receipt, dict)
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt.get("configuration_sha256") != configuration
        or receipt.get("url") != _URL
        or receipt.get("method") != "POST"
        or _canonical(receipt.get("parameters")) != _canonical(_parameters(task, issuer))
        or type(receipt.get("status")) is not int
        or not 100 <= receipt["status"] <= 599
        or type(receipt.get("bytes")) is not int
        or not 0 <= receipt["bytes"] <= _MAX_RESPONSE
        or type(receipt.get("truncated")) is not bool
        or receipt.get("transport") not in {"injected", "urllib_public"}
    ):
        raise ValueError("La respuesta no conserva el contrato de la petición")
    path = directory / "body.json"
    safe_destination(path)
    if not path.is_file() or path.stat().st_size != receipt["bytes"]:
        raise ValueError("La respuesta cambió de tamaño")
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != _hash(receipt.get("sha256")):
        raise ValueError("Cambió el contenido de una respuesta confirmada")
    incomplete = _truncated(receipt.get("headers"), receipt.get("read_bytes"), len(body))
    for file, digest in ((directory / "receipt.json", signature), (path, receipt["sha256"])):
        if file in sources and sources[file] != digest:
            raise ValueError("Una respuesta cambió entre lecturas")
        sources[file] = digest
    effective = {**receipt, "truncated": receipt["truncated"] or incomplete}
    return effective, body, dict(key=_key(task), receipt_sha256=signature)


def _notice(row, task, census):
    if not isinstance(row, dict):
        raise ValueError("El anuncio no es un registro")
    for name, pattern in (
        ("secCode", r"[0-9]{6}"),
        ("announcementId", r"[0-9]{1,32}"),
        ("pageColumn", r"[A-Z0-9_]{1,16}"),
    ):
        if not isinstance(row.get(name), str) or not re.fullmatch(pattern, row[name]):
            raise ValueError("El anuncio no identifica código, mercado o ID")
    stamp = row.get("announcementTime")
    if type(stamp) is not int or not 0 <= stamp <= 4_102_444_800_000:
        raise ValueError("El anuncio no declara una fecha válida")
    day = datetime.fromtimestamp(stamp / 1000, UTC).astimezone(ZoneInfo("Asia/Shanghai")).date()
    if not task["start"] <= day.isoformat() <= task["end"] or day > _CUTOFF:
        raise ValueError("El anuncio no pertenece a la ventana de publicaciones")
    title = row.get("announcementTitle")
    if not isinstance(title, str) or not 1 <= len(title) <= 4096:
        raise ValueError("El anuncio no conserva un título acotado")
    for name in ("secName", "orgId"):
        if row.get(name) is not None and (not isinstance(row[name], str) or len(row[name]) > 256):
            raise ValueError("El emisor no tiene metadatos textuales acotados")
    adjunct = row.get("adjunctUrl")
    if not isinstance(adjunct, str):
        raise ValueError("El enlace del documento debe ser textual")
    match = re.fullmatch(
        r"finalpage/([0-9]{4}-[0-9]{2}-[0-9]{2})/([0-9]+)\.[Pp][Dd][Ff]", adjunct or ""
    )
    if (
        row.get("adjunctType") != "PDF"
        or not match
        or match[1] != day.isoformat()
        or match[2] != row["announcementId"]
    ):
        raise ValueError("El PDF primario no concuerda con la fecha y el ID del anuncio")
    board = row["pageColumn"]
    suffix = (
        ".SZ"
        if board.startswith("SZ")
        else ".SS"
        if board.startswith("SH")
        else ".BJ"
        if board == "BJS"
        else None
    )
    symbol = row["secCode"] + suffix if suffix else None
    return dict(
        announcement_id=row["announcementId"],
        sec_code=row["secCode"],
        symbol=symbol,
        sec_name=row.get("secName"),
        org_id=row.get("orgId"),
        page_column=board,
        publication_date=day.isoformat(),
        announcement_time_ms=stamp,
        title=html.unescape(re.sub(r"<[^>]+>", "", title)),
        pdf_url="https://static.cninfo.com.cn/" + adjunct,
        in_census=symbol in census,
        queue_periods=census.get(symbol, []),
    )


def _issuer(symbol, receipt_path, receipt_hash, announcement_id, census, output, sources):
    values = (symbol, receipt_path, receipt_hash, announcement_id)
    if all(value is None for value in values):
        return None
    if any(value is None for value in values) or symbol not in census:
        raise ValueError("El emisor necesita símbolo del censo, recibo, huella e ID del anuncio")
    path = Path(receipt_path)
    if path.name != "receipt.json":
        raise ValueError("La evidencia necesita receipt.json junto a su body.json conservado")
    outside_source(path.parent, output)
    outside_source(output, path.parent)
    receipt, signature = _json(path, 64 * 1024)
    if signature != _hash(receipt_hash) or not isinstance(receipt, dict):
        raise ValueError("La evidencia del emisor no conserva la huella suministrada")
    parameters = receipt.get("parameters")
    if not isinstance(parameters, dict) or not isinstance(parameters.get("seDate"), str):
        raise ValueError("La evidencia no conserva su ventana de publicación")
    dates = parameters["seDate"].split("~")
    if len(dates) != 2 or _day(dates[0]) > _day(dates[1]):
        raise ValueError("La ventana de la evidencia no es válida")
    page = parameters.get("pageNum")
    if (
        not isinstance(page, str)
        or not re.fullmatch(r"[1-9][0-9]{0,2}", page)
        or int(page) > _MAX_PAGES
    ):
        raise ValueError("La evidencia no conserva una página acotada")
    task = _task(*dates)
    task.update(page=int(page), collected=(int(page) - 1) * _PAGE_SIZE)
    body_path = path.with_name("body.json")
    value, _ = _json(body_path, _MAX_RESPONSE)
    model = _Catalogue(*dates, census)
    model.pending[0] = task.copy()
    model._page(value, _hash(receipt.get("sha256")))
    matches = [
        _notice(row, task, census)
        for row in value["announcements"] or []
        if row["announcementId"] == announcement_id
    ]
    if len(matches) != 1 or matches[0]["symbol"] != symbol:
        raise ValueError("El anuncio seleccionado no identifica el emisor solicitado")
    notice = matches[0]
    if not isinstance(notice["org_id"], str) or not re.fullmatch(
        r"[A-Za-z0-9_-]{1,128}", notice["org_id"]
    ):
        raise ValueError("El anuncio no acredita un orgId individual válido")
    issuer = dict(symbol=symbol, org_id=notice["org_id"])
    checked, _, _ = _response(
        path.parent,
        task,
        _hash(receipt.get("configuration_sha256")),
        sources,
        receipt_hash,
        issuer=issuer if parameters.get("stock") else None,
    )
    if checked["status"] != 200 or checked["truncated"] or checked["transport"] != "urllib_public":
        raise ValueError("La identidad necesita una respuesta pública completa con HTTP 200")
    return dict(
        **issuer,
        announcement_id=announcement_id,
        publication_date=notice["publication_date"],
        announcement=notice,
        category=parameters["category"],
        receipt_path=str(path.resolve()),
        receipt_sha256=signature,
        body_path=str(body_path.resolve()),
        body_sha256=checked["sha256"],
        source_configuration_sha256=checked["configuration_sha256"],
    )


class _Catalogue:
    """Estado derivado de las respuestas confirmadas y del árbol de intervalos."""

    def __init__(self, start, end, census, issuer=None):
        self.pending = [_task(start, end)]
        self.census = census
        self.issuer = issuer
        self.known_announcement = (
            issuer["announcement_id"]
            if issuer and start <= issuer["publication_date"] <= end
            else None
        )
        self.responses = []
        self.records = {}
        self.signatures = {}
        self.completed = {}
        self.parents = {}
        self.splits = {}
        self.failure = None
        self.size = 0

    @property
    def status(self):
        return "blocked" if self.failure else "partial" if self.pending else "completed"

    def cursor(self, configuration):
        return dict(
            schema_version=1,
            configuration_sha256=configuration,
            responses=self.responses,
            pending=self.pending,
            status=self.status,
            announcements=len(self.records),
            failure=self.failure,
        )

    def consume(self, receipt, body):
        if receipt["status"] != 200:
            self.failure = f"CNINFO respondió HTTP {receipt['status']}. No se reintenta."
        elif receipt["truncated"]:
            self.failure = "La respuesta supera el límite de bytes o no acredita estar completa"
        else:
            try:
                self._page(json.loads(body, object_pairs_hook=_unique), receipt["sha256"])
            except (ValueError, OverflowError) as error:
                self.failure = str(error)

    def _page(self, value, source_hash):
        task = self.pending[0]
        total = value.get("totalRecordNum") if isinstance(value, dict) else None
        if (
            type(total) is not int
            or not 0 <= total <= _MAX_RECORDS
            or type(value.get("totalAnnouncement")) is not int
            or value["totalAnnouncement"] != total
            or type(value.get("hasMore")) is not bool
            or type(value.get("totalpages")) is not int
            or value["totalpages"] < 0
            or value.get("actionErrors")
            or value.get("fieldErrors")
            or value.get("error")
        ):
            raise ValueError("Los totales o el estado de la respuesta no son válidos")
        raw = value.get("announcements")
        raw = [] if raw is None and total == 0 else raw
        expected = min(_PAGE_SIZE, max(0, total - (task["page"] - 1) * _PAGE_SIZE))
        if (
            not isinstance(raw, list)
            or len(raw) != expected
            or value["hasMore"] != (task["page"] * _PAGE_SIZE < total)
            or task["total"] is not None
            and task["total"] != total
        ):
            raise ValueError("La población, los totales o hasMore cambiaron entre páginas")
        records = [_notice(row, task, self.census) for row in raw]
        if self.issuer and any(
            row["symbol"] != self.issuer["symbol"] or row["org_id"] != self.issuer["org_id"]
            for row in records
        ):
            raise ValueError("La respuesta no conserva el emisor y orgId solicitados")
        if self.issuer and any(
            row["announcement_id"] == self.issuer["announcement_id"]
            and row != self.issuer["announcement"]
            for row in records
        ):
            raise ValueError("El anuncio de referencia cambió sus metadatos normalizados")
        signatures = {
            r["announcement_id"]: hashlib.sha256(_canonical(r).encode()).hexdigest()
            for r in records
        }
        if len(signatures) != len(records):
            raise ValueError("Hay IDs repetidos en una página")
        interval = (task["start"], task["end"])
        if total > _PAGE_SIZE * _MAX_PAGES:
            if task["start"] == task["end"]:
                raise ValueError("Un solo día supera el límite de páginas de la interfaz")
            first, last = _day(task["start"]), _day(task["end"])
            middle = first + (last - first) // 2
            children = [
                (first.isoformat(), middle.isoformat()),
                ((middle + timedelta(days=1)).isoformat(), last.isoformat()),
            ]
            self.splits[interval] = dict(total=total, children=children, signatures=signatures)
            self.parents.update({child: interval for child in children})
            self.pending[:1] = [_task(*child) for child in children]
            return
        if any(identifier in self.records for identifier in signatures):
            raise ValueError("Un ID se repite entre páginas o intervalos")
        size = sum(len(_canonical(record).encode()) for record in records)
        if len(self.records) + len(records) > _MAX_RECORDS or self.size + size > _MAX_BYTES:
            raise ValueError("El catálogo supera el presupuesto de población o memoria")
        self.size += size
        self.signatures.update(signatures)
        self.records.update(
            {
                r["announcement_id"]: dict(
                    **r,
                    source_response_sha256=source_hash,
                    source_row=i,
                )
                for i, r in enumerate(records, 1)
            }
        )
        task["total"], task["collected"] = total, task["collected"] + len(records)
        if task["collected"] == total:
            self.pending.pop(0)
            self._finish(interval, total)
        else:
            task["page"] += 1

    def _finish(self, interval, total):
        self.completed[interval] = total
        while interval in self.parents:
            interval = self.parents[interval]
            split = self.splits[interval]
            if not all(child in self.completed for child in split["children"]):
                return
            if sum(self.completed[c] for c in split["children"]) != split["total"] or any(
                self.signatures.get(identifier) != digest
                for identifier, digest in split["signatures"].items()
            ):
                raise ValueError("Los subintervalos no concilian con la población y anuncios padre")
            self.completed[interval] = split["total"]
        if not self.pending and self.known_announcement is not None:
            if self.known_announcement not in self.records:
                raise ValueError("Falta el anuncio conocido dentro de la ventana consultada")


def _verify(sources):
    for path, signature in sources.items():
        safe_destination(path)
        if not path.is_file() or sha256(path) != signature:
            raise ValueError("Una fuente o archivo confirmado cambió durante el catálogo")


def _write_cursor(output, model, configuration, sources):
    path = output / "cursor.json"
    value = model.cursor(configuration)
    atomic_json(path, value)
    actual, signature = _json(path)
    if _canonical(actual) != _canonical(value):
        raise ValueError("El cursor cambió durante la confirmación")
    sources[path] = signature


def _replay(output, configuration, start, end, census, sources, issuer=None):
    model = _Catalogue(start, end, census, issuer)
    path = output / "cursor.json"
    if not path.exists():
        responses = output / "responses"
        safe_destination(responses)
        if responses.exists() and any(responses.iterdir()):
            raise ValueError("Falta el cursor de las respuestas ya conservadas")
        _write_cursor(output, model, configuration, sources)
        return model
    cursor, sources[path] = _json(path)
    references = cursor.get("responses") if isinstance(cursor, dict) else None
    if not isinstance(references, list) or len(references) > _MAX_RESPONSES:
        raise ValueError("El cursor no conserva un historial acotado")
    for reference in references:
        if (
            model.status != "partial"
            or not isinstance(reference, dict)
            or set(reference) != {"key", "receipt_sha256"}
            or reference["key"] != _key(model.pending[0])
        ):
            raise ValueError("El cursor no corresponde al siguiente intervalo y página")
        receipt, body, actual = _response(
            output / "responses" / reference["key"],
            model.pending[0],
            configuration,
            sources,
            _hash(reference["receipt_sha256"]),
            issuer=issuer,
        )
        model.consume(receipt, body)
        model.responses.append(actual)
    if _canonical(cursor) != _canonical(model.cursor(configuration)):
        raise ValueError("El cursor no conserva el estado y tipos derivados de las respuestas")
    return model


def _report(model, configuration, config, output, sources):
    report = dict(
        schema_version=1,
        kind="cn_annual_announcement_catalogue",
        status=model.status,
        configuration_sha256=configuration,
        cursor_sha256=sources[output / "cursor.json"],
        source_url=_URL,
        category="category_ndbg_szsh",
        searchkey="",
        publication_start=config["publication_start"],
        publication_end=config["publication_end"],
        confirmed_requests=len(model.responses),
        announcements=len(model.records),
        in_census=sum(r["in_census"] for r in model.records.values()),
        queue_rows=config["queue_rows"],
        queue_symbols=len(model.census),
        queue_periods=config["queue_periods"],
        period_coverage_verified=False,
        financial_values_admitted=False,
        training_ready=False,
        final_test_opened=False,
        failure=model.failure,
        scope="annual_category_only",
    )
    if model.issuer:
        report.update(
            schema_version=2,
            scope="issuer_annual_category_only",
            issuer=model.issuer,
            known_announcement_required=model.known_announcement is not None,
            known_announcement_present=model.known_announcement in model.records
            if model.known_announcement is not None
            else None,
            issuer_history_complete=False,
        )
    if model.status == "partial":
        return report
    path = output / "report.json"
    previous = _json(path)[0] if path.exists() else None
    if model.status == "completed":
        rows = [model.records[k] for k in sorted(model.records)]
        table = pa.Table.from_pylist(rows, schema=_SCHEMA)
        target = output / "announcements.parquet"
        safe_destination(target)
        if target.exists():
            actual = read_bounded_table(target, max_rows=_MAX_RECORDS, max_bytes=_MAX_BYTES)
            if not actual.equals(table):
                raise ValueError("El catálogo confirmado no coincide con las respuestas")
        elif previous is not None:
            raise ValueError("Falta el catálogo de una colección confirmada")
        else:
            atomic_parquet(target, table)
        report["artifacts"] = {"announcements.parquet": sha256(target)}
    _verify(sources)
    if previous is not None:
        if _canonical(previous) != _canonical(report):
            raise ValueError("El informe no conserva el contenido y tipos confirmados")
    else:
        atomic_json(path, report)
    return report


def collect_chinese_announcements(
    queue,
    output,
    *,
    publication_start,
    publication_end,
    max_requests,
    issuer_symbol=None,
    issuer_receipt=None,
    issuer_receipt_sha256=None,
    issuer_announcement_id=None,
    transport=None,
    sleep=time.sleep,
):
    """Recoger metadatos anuales. El transporte inyectado devuelve estado, cabeceras y bytes."""
    start, end = _day(publication_start), _day(publication_end)
    if start > end or type(max_requests) is not int or not 1 <= max_requests <= 1000:
        raise ValueError("La ventana o el presupuesto de peticiones no son válidos")
    if not callable(sleep) or transport is not None and not callable(transport):
        raise ValueError("El transporte y la espera deben ser funciones")
    queue, output = Path(queue), Path(output)
    safe_destination(output)
    for source in (queue.parent, Path("dataset")):
        outside_source(source, output)
        outside_source(output, source)
    census, queue_hash, count = _queue(queue)
    sources = {queue: queue_hash}
    code_sources = {
        Path(__file__).with_name(name): sha256(Path(__file__).with_name(name)) for name in _CODE
    }
    sources.update(code_sources)
    issuer = _issuer(
        issuer_symbol,
        issuer_receipt,
        issuer_receipt_sha256,
        issuer_announcement_id,
        census,
        output,
        sources,
    )
    config = dict(
        schema_version=1,
        queue_path=str(queue.resolve()),
        queue_sha256=queue_hash,
        queue_rows=count,
        queue_symbols=sorted(census),
        queue_periods=sorted({period for periods in census.values() for period in periods}),
        publication_start=start.isoformat(),
        publication_end=end.isoformat(),
        source_url=_URL,
        category="category_ndbg_szsh",
        searchkey="",
        scope="annual_category_only",
        period_coverage_verified=False,
        code={p.name: d for p, d in code_sources.items()},
        limits=dict(
            page_size=_PAGE_SIZE,
            pages_per_interval=_MAX_PAGES,
            response_bytes=_MAX_RESPONSE,
            timeout_seconds=30,
            request_spacing_seconds=2,
            total_records=_MAX_RECORDS,
            confirmed_responses=_MAX_RESPONSES,
        ),
    )
    if issuer:
        config.update(schema_version=2, scope="issuer_annual_category_only", issuer=issuer)
        config["limits"]["request_spacing_seconds"] = 5
    output.mkdir(parents=True, exist_ok=True)
    lock_path = output / ".catalogue.lock"
    safe_destination(lock_path)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        path = output / "configuration.json"
        if not path.exists():
            if any(p.name != lock_path.name for p in output.iterdir()):
                raise ValueError("El destino contiene una colección sin configuración")
            atomic_json(path, config)
        actual, configuration = _json(path)
        if _canonical(actual) != _canonical(config):
            raise ValueError("La configuración pertenece a otra cola, código o ventana")
        sources[path] = configuration
        terminal = output / "report.json"
        if terminal.exists():
            previous, sources[terminal] = _json(terminal)
            cursor = output / "cursor.json"
            safe_destination(cursor)
            if (
                not isinstance(previous, dict)
                or previous.get("configuration_sha256") != configuration
                or previous.get("status") not in {"completed", "blocked"}
                or not cursor.is_file()
                or sha256(cursor) != _hash(previous.get("cursor_sha256"))
            ):
                raise ValueError("El informe terminal no conserva su cursor confirmado")
        model = _replay(
            output, configuration, start.isoformat(), end.isoformat(), census, sources, issuer
        )
        used = 0
        while model.status == "partial":
            task = model.pending[0]
            directory = output / "responses" / _key(task)
            safe_destination(directory)
            if not directory.exists():
                if used == max_requests:
                    break
                if len(model.responses) >= _MAX_RESPONSES:
                    raise ValueError("El historial supera el presupuesto de respuestas")
                _verify(sources)
                sleep(config["limits"]["request_spacing_seconds"])
                requested_at = datetime.now(UTC).isoformat()
                used += 1
                try:
                    response = (transport or _request)(_parameters(task, issuer))
                except OSError as error:
                    atomic_json(
                        output / f"transport-error-{time.time_ns()}.json",
                        dict(
                            requested_at_utc=requested_at,
                            parameters=_parameters(task, issuer),
                            error=str(error),
                        ),
                    )
                    raise
                _save_response(
                    directory,
                    task,
                    configuration,
                    response,
                    requested_at,
                    transport is not None,
                    issuer,
                )
            receipt, body, reference = _response(
                directory, task, configuration, sources, issuer=issuer
            )
            model.consume(receipt, body)
            model.responses.append(reference)
            _verify(sources)
            _write_cursor(output, model, configuration, sources)
        _verify(sources)
        report = _report(model, configuration, config, output, sources)
    return {**report, "requests_this_invocation": used}


def main(argv=None, *, transport=None, sleep=time.sleep):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--publication-start", required=True)
    parser.add_argument("--publication-end", required=True)
    parser.add_argument("--max-requests", type=int, required=True)
    parser.add_argument("--issuer-symbol")
    parser.add_argument("--issuer-receipt", type=Path)
    parser.add_argument("--issuer-receipt-sha256")
    parser.add_argument("--issuer-announcement-id")
    args = parser.parse_args(argv)
    report = collect_chinese_announcements(**vars(args), transport=transport, sleep=sleep)
    print(f"Catálogo: {report['status']}. Anuncios: {report['announcements']}.")
    return 2 if report["status"] == "blocked" else 0


if __name__ == "__main__":
    raise SystemExit(main())
