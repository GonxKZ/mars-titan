"""Extraer cifras de comunicados oficiales conservando periodo y fechas declaradas."""

import calendar
import copy
import hashlib
import math
import re
import unicodedata
from datetime import date
from decimal import Decimal, InvalidOperation
from io import StringIO

import pandas as pd
from bs4 import BeautifulSoup

from .macro_release_contracts import release_source

_MONTHS = (
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December",
)
_MONTH_NUMBERS = {name.lower(): i for i, name in enumerate(_MONTHS, 1)}


def _text(value):
    return "" if pd.isna(value) else " ".join(unicodedata.normalize("NFKC", str(value)).split())


def _key(value):
    return re.sub(r"[^a-z0-9]", "", _text(value).lower())


def _number(value):
    text = _text(value).replace("−", "-").removesuffix("%")
    if len(text) > 64:
        raise ValueError("La cifra publicada supera el presupuesto de texto")
    try:
        number = Decimal(text)
    except InvalidOperation as error:
        raise ValueError("El comunicado no contiene una cifra inequívoca") from error
    if not number.is_finite() or not math.isfinite(float(number)):
        raise ValueError("La cifra publicada no es finita")
    if number != 0 and float(number) == 0:
        raise ValueError("La cifra publicada no puede representarse sin perder su magnitud")
    return float(number)


def _tables(body):
    for table in body.find_all("table"):
        cells = table.find_all(("td", "th"))
        if len(cells) > 5000:
            raise ValueError("La tabla del comunicado supera el presupuesto")
        expanded = 0
        for cell in cells:
            try:
                rows, columns = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            except ValueError as error:
                raise ValueError("La tabla contiene combinaciones de celdas no válidas") from error
            if not 1 <= rows <= 32 or not 1 <= columns <= 32:
                raise ValueError("La expansión de celdas supera el presupuesto")
            expanded += rows * columns
        if expanded > 50_000:
            raise ValueError("La tabla expandida supera el presupuesto")
        # Algunas publicaciones incluyen todas las cifras dentro de thead.
        normalized = copy.deepcopy(table)
        for node in normalized.find_all(("thead", "th")):
            node.name = "tbody" if node.name == "thead" else "td"
        frames = pd.read_html(StringIO(str(normalized)), flavor="lxml", header=None)
        if len(frames) != 1 or frames[0].size > 50_000:
            raise ValueError("La tabla no tiene una estructura única y acotada")
        yield table, frames[0]


def _yoy_columns(frame):
    return {
        column
        for column in range(frame.shape[1])
        if any(
            "y/y" in _text(frame.iat[row, column]).lower()
            or "yearonyear" in _key(frame.iat[row, column])
            for row in range(min(3, len(frame)))
        )
    }


def _reference(title):
    years = set(re.findall(r"\b(20\d{2})\b", title))
    if len(years) != 1:
        raise ValueError("El comunicado necesita un año de referencia inequívoco")
    months = {
        number
        for name, number in _MONTH_NUMBERS.items()
        if re.search(r"\b" + name + r"\b", title.lower())
    }
    return int(next(iter(years))), months


def _record(common, indicator, period, value, adjustment):
    last = date(period.year, period.month, calendar.monthrange(period.year, period.month)[1])
    if last.isoformat() > common["declared_publication_date"]:
        raise ValueError("El periodo del comunicado termina después de su publicación")
    unit = "diffusion_index_50_neutral" if indicator == "cn_manufacturing_pmi" else "percent_yoy"
    return dict(
        common,
        indicator_id=indicator,
        period_start=period.isoformat(),
        reference_period_end=last.isoformat(),
        value=value,
        native_unit=unit,
        seasonal_adjustment=adjustment,
        missing_reason=None,
    )


def _cpi_rows(common, body, year, months):
    records = []
    if len(months) != 1:
        raise ValueError("El IPC requiere un único mes de referencia")
    period = date(year, next(iter(months)), 1)
    for _, frame in _tables(body):
        targets = [
            i
            for i in range(len(frame))
            if _key(frame.iat[i, 0])
            in {"consumerprices", "consumerpriceindex", "consumerpriceindexcpi"}
        ]
        if not targets:
            continue
        columns = _yoy_columns(frame)
        if len(columns) != 1 or len(targets) != 1:
            raise ValueError("La tabla no distingue un único IPC interanual")
        value = _number(frame.iat[targets[0], next(iter(columns))])
        records.append(_record(common, "cn_cpi_yoy_published", period, value, "published_yoy"))
    return records


def _industry_rows(common, body, year, months):
    records = []
    for _, frame in _tables(body):
        targets = [
            i
            for i in range(len(frame))
            if _key(frame.iat[i, 0]).startswith("valueaddedofindustr")
            and "designatedsize" in _key(frame.iat[i, 0])
        ]
        if not targets:
            continue
        candidates = [
            (column, _MONTH_NUMBERS[_text(frame.iat[0, column]).lower()])
            for column in _yoy_columns(frame)
            if _text(frame.iat[0, column]).lower() in _MONTH_NUMBERS
        ]
        if not candidates and months == {1, 2}:
            return None
        if len(candidates) != 1 or len(targets) != 1:
            raise ValueError("La tabla no distingue el dato industrial mensual del acumulado")
        column, month = candidates[0]
        if months and month not in months:
            raise ValueError("El mes de la tabla no corresponde al título")
        records.append(
            _record(
                common,
                "cn_industrial_yoy_published",
                date(year, month, 1),
                _number(frame.iat[targets[0], column]),
                "published_real_yoy",
            )
        )
    return records


def _validate_pmi_adjustment(table, frame):
    labels = {_key(value) for value in frame.iloc[:4].to_numpy().flat}
    for node in table.find_all_previous(("p", "table"), limit=6):
        if node.name == "table" or node.find_parent("table") is not None:
            break
        labels.add(_key(node.get_text(" ", strip=True)))
    declarations = set()
    for label in labels:
        match = re.fullmatch(r"(?:chinas)?((?:non)?manufacturing)pmi(.*)", label)
        if match:
            declarations.add(match.groups())
    if declarations != {("manufacturing", "seasonallyadjusted")}:
        raise ValueError("No consta un ajuste inequívoco del PMI manufacturero publicado")


def _pmi_rows(common, body, year, months):
    records = []
    if len(months) != 1:
        raise ValueError("El PMI requiere un único mes de referencia")
    target = date(year, next(iter(months)), 1)
    for table, frame in _tables(body):
        columns = {
            column
            for column in range(frame.shape[1])
            if any(_key(frame.iat[row, column]) == "pmi" for row in range(min(4, len(frame))))
        }
        if not columns:
            continue
        if len(columns) != 1:
            raise ValueError("La tabla mezcla varios índices PMI")
        _validate_pmi_adjustment(table, frame)
        column, current_year, previous = next(iter(columns)), None, None
        for row in range(len(frame)):
            text = _text(frame.iat[row, 0])
            match = re.fullmatch(r"(?:(20\d{2})[-–\s]+)?([A-Za-z]+)", text)
            if not match or match.group(2).lower() not in _MONTH_NUMBERS:
                continue
            current_year = int(match.group(1)) if match.group(1) else current_year
            if current_year is None:
                raise ValueError("El historial PMI no identifica su primer año")
            period = date(current_year, _MONTH_NUMBERS[match.group(2).lower()], 1)
            if previous is not None and period <= previous or period > target:
                raise ValueError("El historial PMI está desordenado o contiene un periodo futuro")
            previous = period
            value = _number(frame.iat[row, column])
            if not 0 <= value <= 100:
                raise ValueError("El PMI debe estar entre cero y cien")
            records.append(
                _record(common, "cn_manufacturing_pmi", period, value, "Seasonally Adjusted")
            )
        if previous != target:
            raise ValueError("El PMI no contiene el mes anunciado")
    return records


def extract_nbs_release(content: bytes, source_url: str) -> dict:
    """Leer las tablas nacionales, distinguiendo dato mensual y acumulado.

    La disponibilidad usa el día más tardío entre el declarado y el de la ruta
    archivada. Ambos se conservan. El límite no se presenta como una hora exacta.
    """
    family, path_day = release_source(source_url)
    if family != "nbs":
        raise ValueError("Se necesita un comunicado HTTPS del archivo oficial de NBS")
    if not isinstance(content, bytes) or not 0 < len(content) <= 3 * 1024**2:
        raise ValueError("El comunicado está vacío o supera tres MiB")
    soup = BeautifulSoup(content, "lxml")
    title_node, info, body = (
        soup.select_one("h1.con_titles"),
        soup.select_one("div.info"),
        soup.select_one(".TRS_Editor"),
    )
    if any(node is None for node in (title_node, info, body)):
        raise ValueError("Faltan título, fecha o cuerpo del comunicado NBS")
    title = title_node.get_text(" ", strip=True)
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", info.get_text(" ", strip=True)))
    if len(dates) != 1 or "National Bureau of Statistics" not in info.get_text(" ", strip=True):
        raise ValueError("La fecha o procedencia del comunicado NBS no es inequívoca")
    publication = date.fromisoformat(next(iter(dates)))
    available = max(publication, path_day)
    common = dict(
        source_url=source_url,
        source_title=title,
        source_hash=hashlib.sha256(content).hexdigest(),
        source_timezone="Asia/Shanghai",
        declared_publication_date=publication.isoformat(),
        archive_path_date=path_day.isoformat(),
        realtime_start=available.isoformat(),
        realtime_end="9999-12-31",
        availability_policy="latest_declared_or_archive_day_then_next_session",
        publication_timestamp_verified=False,
    )
    year, months = _reference(title)
    records = []
    label = _key(title)
    if label.startswith("consumerprice"):
        records = _cpi_rows(common, body, year, months)
    elif label.startswith("industrialproductionoperation"):
        records = _industry_rows(common, body, year, months)
        if records is None:
            return dict(common, observations=[], exclusion="nonmonthly_reference_period")
    elif "purchasingmanagers" in label:
        records = _pmi_rows(common, body, year, months)
    else:
        raise ValueError("El título no corresponde a un indicador NBS admitido")
    if not records:
        raise ValueError("No se encontró la cifra solicitada en sus tablas oficiales")
    keys = [(row["indicator_id"], row["period_start"]) for row in records]
    if len(set(keys)) != len(keys):
        raise ValueError("El comunicado repite observaciones incompatibles")
    return dict(common, observations=records, exclusion=None)


def _financial_period(title):
    match = re.fullmatch(
        r"(20\d{2})年([0-9]+月|一季度|上半年|前三季度)?"
        r"(金融|社会融资规模[增存]量)统计数据报告",
        title,
    )
    if not match:
        raise ValueError("El título no identifica un informe financiero nacional admitido")
    label = match.group(2)
    quarter = {None: 12, "一季度": 3, "上半年": 6, "前三季度": 9}
    month = quarter[label] if label in quarter else int(label[:-1])
    return date(int(match.group(1)), month, 1), match.group(3)


def extract_pboc_release(content: bytes, source_url: str) -> dict:
    """Leer saldos y flujos nacionales publicados por PBoC o su reproducción oficial.

    Las reproducciones usan su propia fecha. Los acumulados trimestrales o
    anuales nunca se convierten por diferencia en un flujo mensual.
    """
    family, _ = release_source(source_url)
    original, mirror = family == "pboc", family == "pboc_reprint"
    if not (original or mirror):
        raise ValueError("Se necesita un comunicado del archivo financiero oficial admitido")
    if not isinstance(content, bytes) or not 0 < len(content) <= 3 * 1024**2:
        raise ValueError("El comunicado está vacío o supera tres MiB")
    soup = BeautifulSoup(content, "lxml")
    title_node = soup.select_one("h2" if original else ".title_content")
    published = soup.select_one("#shijian" if original else ".c_time")
    body = soup.select_one("#zoom")
    if any(node is None for node in (title_node, published, body)):
        raise ValueError("Faltan título, fecha o cuerpo del comunicado financiero")
    if mirror:
        origin = soup.select_one("#ly")
        if origin is None or origin.get_text(strip=True) != "中国人民银行":
            raise ValueError("La reproducción no identifica al Banco Popular de China")
    title = "".join(_text(title_node.get_text()).split())
    period, kind = _financial_period(title)
    dates = set(re.findall(r"\b\d{4}-\d{2}-\d{2}\b", published.get_text()))
    if len(dates) != 1:
        raise ValueError("El comunicado no tiene un día de publicación inequívoco")
    publication = date.fromisoformat(next(iter(dates)))
    meta = soup.select_one('meta[name="PubDate"]')
    if original and (meta is None or meta.get("content") != publication.isoformat()):
        raise ValueError("Las fechas de publicación del comunicado no coinciden")
    last = date(period.year, period.month, calendar.monthrange(period.year, period.month)[1])
    if last > publication:
        raise ValueError("El periodo financiero termina después de la publicación")
    common = dict(
        source_url=source_url,
        source_title=title,
        source_hash=hashlib.sha256(content).hexdigest(),
        source_timezone="Asia/Shanghai",
        declared_publication_date=publication.isoformat(),
        realtime_start=publication.isoformat(),
        realtime_end="9999-12-31",
        availability_policy="declared_publication_day_then_next_session",
        publication_timestamp_verified=False,
    )
    text = "".join(_text(body.get_text(" ", strip=True)).split())
    month = rf"(?<![\d年\-])(?:{period.year}年)?{period.month}月"
    if kind == "金融":
        indicator = "cn_m2_stock"
        pattern = month + r"末[,，]?广义货币\(M2\)余额"
    elif kind == "社会融资规模存量":
        indicator = "cn_tsf_stock"
        prefix = rf"(?:{month}末|{period.year}年末)" if period.month == 12 else month + "末"
        pattern = prefix + r"[,，]?社会融资规模存量为"
    else:
        indicator = "cn_tsf_flow"
        pattern = month + r"份?[,，]?社会融资规模增量为"
        if not re.search(pattern, text) and "社会融资规模增量累计为" in text:
            return dict(common, observations=[], exclusion="monthly_flow_not_published")
    matches = list(re.finditer(pattern + r"(-?\d{1,16}(?:\.\d{1,10})?)(万亿元|亿元)", text))
    if len(matches) != 1:
        raise ValueError("El comunicado no contiene una única cifra mensual del periodo declarado")
    match = matches[0]
    amount, unit = Decimal(match.group(1)), match.group(2)
    if indicator != "cn_tsf_flow" and amount <= 0:
        raise ValueError("Los saldos monetarios deben ser positivos")
    multiplier = Decimal(1000) if unit == "万亿元" else Decimal("0.1")
    normal_unit = (
        "billion_CNY_per_month_normalized_from_release"
        if indicator == "cn_tsf_flow"
        else "billion_CNY_normalized_from_release"
    )
    row = dict(
        common,
        indicator_id=indicator,
        period_start=period.isoformat(),
        reference_period_end=last.isoformat(),
        value=float(amount * multiplier),
        source_value=str(amount),
        source_unit=unit,
        native_unit=normal_unit,
        seasonal_adjustment="published_without_adjustment_statement",
        missing_reason=None,
    )
    return dict(common, observations=[row], exclusion=None)
