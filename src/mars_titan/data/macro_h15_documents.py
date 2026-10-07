"""Extraer candidatos diarios H.15 sin alterar decimales ni atribuir una revisión."""

import hashlib
import math
import re
from datetime import date
from decimal import Decimal, InvalidOperation
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from .macro_release_documents import _MONTH_NUMBERS, _text

_MAX_DOCUMENT_BYTES = 8 * 1024**2


_SERIES = {
    "Federal funds (effective)": ("us_fed_funds", "DFF"),
    "3-month": ("us_treasury_3m", "DGS3MO"),
    "2-year": ("us_treasury_2y", "DGS2"),
    "5-year": ("us_treasury_5y", "DGS5"),
    "10-year": ("us_treasury_10y", "DGS10"),
    "30-year": ("us_treasury_30y", "DGS30"),
}


_MONTHS = _MONTH_NUMBERS | {name[:3]: n for name, n in _MONTH_NUMBERS.items()}


def _day(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("La fecha debe tener formato ISO completo")
    result = date.fromisoformat(value)
    if not date(1900, 1, 1) <= result < date(2024, 1, 1):
        raise ValueError("La fecha queda fuera del archivo histórico o abre la reserva")
    return result


def _english_day(value, *, column=False):
    pattern = r"(\d{4}) ([A-Za-z]+) (\d{1,2})" if column else r"([A-Za-z]+) (\d{1,2}), (\d{4})"
    match = re.fullmatch(pattern, value)
    if match is None:
        raise ValueError("La fecha H.15 no identifica año, mes y día")
    year, month, day = match.groups() if column else (match[3], match[1], match[2])
    if month.lower() not in _MONTHS:
        raise ValueError("El mes del H.15 no está reconocido")
    return _day(f"{int(year):04d}-{_MONTHS[month.lower()]:02d}-{int(day):02d}")


def _issue_url(value, extension):
    if not isinstance(value, str) or len(value) > 1024:
        raise ValueError("La URL de H.15 no es válida")
    parsed = urlsplit(value)
    match = re.fullmatch(r"/releases/[hH]15/(\d{8})/h15\." + extension, parsed.path)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "www.federalreserve.gov"
        or parsed.query
        or parsed.fragment
        or match is None
    ):
        raise ValueError("Se necesita una URL HTTPS de la edición oficial H.15")
    compact = match[1]
    return _day(f"{compact[:4]}-{compact[4:6]}-{compact[6:]}")


def _decimal(text):
    if text in {"", "n.a.", "ND"}:
        return None
    if not isinstance(text, str) or len(text) > 64:
        raise ValueError("La cifra excede su presupuesto textual")
    try:
        number = Decimal(text.replace("−", "-"))
    except InvalidOperation as error:
        raise ValueError("La celda H.15 no contiene una cifra inequívoca") from error
    if not number.is_finite() or not math.isfinite(float(number)):
        raise ValueError("La cifra H.15 no es finita")
    if number != 0 and float(number) == 0:
        raise ValueError("La cifra pierde su magnitud al representarse")
    return str(number)


def _cells(row):
    cells = row.find_all(("td", "th"), recursive=False)
    if len(cells) != 9 or any(
        cell.get("colspan", "1") != "1" or cell.get("rowspan", "1") != "1" for cell in cells
    ):
        raise ValueError("La fila H.15 no contiene nueve celdas independientes")
    return [_text(cell.get_text(" ", strip=True)) for cell in cells]


def extract_h15_release(content: bytes, source_url: str) -> dict:
    """Extraer treinta candidatos diarios. Los promedios no son observaciones diarias."""
    published = _issue_url(source_url, "htm")
    if not isinstance(content, bytes) or not 0 < len(content) <= _MAX_DOCUMENT_BYTES:
        raise ValueError("El HTML vacío o mayor de ocho MiB supera el presupuesto")
    if content.count(b"<") > 50_000:
        raise ValueError("El HTML supera el presupuesto de elementos")
    soup = BeautifulSoup(content, "lxml")
    if len(soup.find_all(("td", "th"))) > 5000:
        raise ValueError("La tabla supera el presupuesto de celdas")
    for node in soup.find_all(("script", "style")):
        node.decompose()
    text = _text(soup.get_text(" ", strip=True))
    declarations = re.findall(
        r"(?:Release Date:|For immediate release)\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", text
    )
    if not declarations or {_english_day(v) for v in declarations} != {published}:
        raise ValueError("La fecha impresa no coincide con la edición de la URL")
    if "Yields in percent per annum" not in text or "H.15" not in text:
        raise ValueError("Falta la identidad H.15 o su unidad porcentual anual")
    tables = [
        t
        for t in soup.find_all("table")
        if _text(t.get("summary", "")) == "Selected Interest Rates"
    ]
    if len(tables) != 1:
        raise ValueError("No se identifica una única tabla H.15")
    table = tables[0]
    rows = [r for r in table.find_all("tr") if r.find_parent("table") is table]
    headers = [
        r
        for r in rows
        if r.find(("td", "th")) is not None
        and _text(r.find(("td", "th")).get_text()) == "Instruments"
    ]
    if len(headers) != 1:
        raise ValueError("Los encabezados diarios no son únicos")
    header = _cells(headers[0])
    days = [_english_day(v, column=True) for v in header[1:6]]
    if days != sorted(set(days)) or any(day > published for day in days):
        raise ValueError("Las columnas diarias están repetidas, desordenadas o son futuras")
    if any(re.fullmatch(r"\d{4} [A-Za-z]+ \d{1,2}", v) for v in header[6:]):
        raise ValueError("Una columna agregada se presenta como observación diaria")
    observations, found, section = [], set(), False
    for index, row in enumerate(rows, 1):
        first = row.find(("td", "th"))
        label = _text(first.get_text(" ", strip=True)) if first is not None else ""
        if re.fullmatch(r"Treasury constant maturities(?: \d+)*", label):
            if section:
                raise ValueError("La sección Treasury está duplicada")
            section = True
            continue
        cells = row.find_all(("td", "th"), recursive=False)
        if label in {"Composite", "Corporate bonds"} or (
            section
            and label
            and re.fullmatch(r"\d+-(?:month|year)", label) is None
            and all(not _text(cell.get_text(" ", strip=True)) for cell in cells[1:])
        ):
            section = False
        name = None
        if re.fullmatch(r"Federal funds \(effective\)(?: \d+)*", label):
            name = "Federal funds (effective)"
        elif section:
            match = re.fullmatch(
                r"(3-month|2-year|5-year|10-year|30-year)(?:\s*;|\s*&nb sp;)?", label
            )
            name = match[1] if match else None
        if name is None:
            continue
        if name in found:
            raise ValueError("La edición contiene filas contradictorias o duplicadas")
        found.add(name)
        cells = _cells(row)
        for column, (day, cell) in enumerate(zip(days, cells[1:6], strict=True), 1):
            value = _decimal(cell)
            observations.append(
                dict(
                    indicator_id=_SERIES[name][0],
                    series_id=_SERIES[name][1],
                    period_start=day.isoformat(),
                    value_exact=value,
                    missing_reason="missing_source_value" if value is None else None,
                    source_row=index,
                    source_column=column,
                    source_label=name,
                )
            )
    if found != set(_SERIES):
        raise ValueError("La edición no acredita las seis series permitidas")
    pdf_url = source_url.removesuffix("htm") + "pdf"
    if not any(urljoin(source_url, node.get("href", "")) == pdf_url for node in soup.find_all("a")):
        raise ValueError("El HTML no enlaza el PDF de la misma edición")
    return dict(
        schema_version=1,
        status="candidate_extracted",
        admission_required=True,
        publication_date=published.isoformat(),
        source_url=source_url,
        pdf_url=pdf_url,
        html_sha256=hashlib.sha256(content).hexdigest(),
        observations=observations,
        unit="percent_per_annum",
        source_timezone="America/New_York",
    )
