"""Tabla del estado de cotización de las acciones A de la edición, desde capturas oficiales.

La tabla que lee `simulation/listing_status.py` necesita, para cada activo chino, la fecha
de admisión, la primera sesión con banda tras una salida a bolsa sin límites, los tramos
con advertencia de riesgo (ST o *ST), los tramos sin reforma accionarial (prefijo S) y los
días sin límite que siguen a una reforma o a una reanudación de cotización. Este módulo la
construye solo con capturas ya guardadas, cada una con su recibo y su huella, sin hacer
ninguna petición:

- Admisión: listas oficiales de Shanghái (acciones A del tablero principal, STAR y
  cotizaciones terminadas) y de Shenzhen (acciones A y valores suspendidos o terminados).
- Tramos de Shenzhen: el informe oficial de cambios de nombre abreviado de la bolsa. Un
  tramo ST empieza cuando el nombre pasa a llevar ST y termina cuando lo pierde, y un tramo
  S hace lo mismo con el prefijo de las acciones sin reforma.
- Tramos ST de Shanghái: la bolsa no publica ese historial, así que se leen los anuncios de
  implantación, cambio y retirada de la advertencia publicados en CNINFO. De cada texto se
  extrae la fecha efectiva y el sentido del cambio. El nombre que CNINFO asocia a cada
  anuncio en su fecha de publicación sirve de observación independiente del estado.
- Días sin límite: los anuncios de ejecución de la reforma accionarial y de reanudación
  de la cotización dicen qué día no tiene banda. En Shanghái, el tramo S de una acción
  reformada después de `COVERAGE_FROM` termina ese mismo día.

El mismo extractor se aplica a los anuncios de Shenzhen y se compara día a día con el
informe oficial. Esa concordancia, junto con un contraste de los tramos con las bandas de
los precios sin ajustar, queda en el informe de la construcción. Cualquier contradicción
desde `COVERAGE_FROM` detiene la construcción en lugar de producir una tabla dudosa.

No se inventan fechas. Un tramo cuyo inicio no consta en los anuncios solo se admite si la
primera evidencia del estado es anterior a `COVERAGE_FROM`, y entonces empieza en esa fecha,
que es lo que la tabla puede afirmar. Las salidas con precio quedan vacías: ninguna fuente
capturada fija un importe de salida para los activos de la edición.
"""

import argparse
import hashlib
import json
import re
import subprocess
import xml.etree.ElementTree as ElementTree
import zipfile
from bisect import bisect_right
from collections import defaultdict
from datetime import UTC, date, datetime, timedelta, timezone
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path

from mars_titan.simulation.listing_status import (
    COVERAGE_FROM,
    CUTOFF,
    TABLE_KIND,
    read_listing_status,
)

from .storage import atomic_json, outside_source
from .temporal import MarketClock

PDFTOTEXT = "/usr/bin/pdftotext"
SPECIAL_BAND = 0.05
MAX_CAPTURE_BYTES = 24 * 1024**2
MAX_TEXT_BYTES = 2 * 1024**2
CHINA = timezone(timedelta(hours=8))
# Nombre con advertencia de riesgo, también con los prefijos S (sin reforma accionarial) o G.
ST_NAME = re.compile(r"^\s*[SG]?\s*\*?\s*S\s*T", re.IGNORECASE)
# Nombre de una acción sin reforma accionarial: S delante de ST, de *ST o del nombre chino.
S_NAME = re.compile(r"^\s*S(?=\s*\*?\s*S\s*T|\s*[^\sA-Za-z*])")
# Títulos que fijan el primer día de una reforma accionarial o de una reanudación.
_NO_LIMIT = re.compile(r"不(?:设|实行)(?:价格)?(?:涨跌|跌涨)幅")
# Un producto que vuelve al mercado (疫苗) también «恢复上市», pero no es la acción.
_NOT_FREE = r"疫苗|产品|申请|债券|子公司|进展|核准|受理|补充|保荐|法律意见|承诺"
_FREE_TITLE = re.compile(
    r"股权分置改革(?:方案)?实施(?:公告|完毕)|实施股权分置改革的公告|恢复上市的公告|恢复上市公告"
    r"|恢复上市暨复牌|恢复上市首日"
)
SSE_LISTS = ("sse-gp-l-common-type1", "sse-gp-l-common-type8", "sse-terminated-main")
SZSE_LISTS = (
    "szse-a-share-list-1110-xlsx",
    "szse-suspended-terminated-1793-tab1-xlsx",
    "szse-suspended-terminated-1793-tab2-xlsx",
)
SZSE_NAMES = "szse-changename-tab2-xlsx"
# Un anuncio fija su fecha efectiva en las semanas siguientes a su publicación.
EFFECTIVE_WINDOW = (-3, 45)
_DATE = re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日")
_OPEN = '[“"「『‘]'
_CLOSE = '[”"」』’]'
_RENAME = re.compile(
    r"简称\s*(?:由\s*"
    + _OPEN
    + r"?\s*([^”\"」』’，,。]{1,12}?)\s*"
    + _CLOSE
    + r"?\s*)?(?:变更为|改为|变为|更改为|更名为)\s*"
    + _OPEN
    + r"\s*([^”\"」』’]{1,12}?)\s*"
    + _CLOSE
)
_TEN = re.compile(
    r"(?:变更|恢复|改|变|调整)为\s*[±+\-]?\s*10\s*[%％]|限制\s*(?:为|是)\s*[±+\-]?\s*10\s*[%％]"
)
_STILL_FIVE = re.compile(r"仍(?:为|是)\s*[±+\-]?\s*5\s*[%％]")
_FIVE = re.compile(
    r"(?:变更|改|变|调整)为\s*[±+\-]?\s*5\s*[%％]|限制\s*(?:为|是)\s*[±+\-]?\s*5\s*[%％]"
)
_EFFECTIVE = re.compile(
    r"(?:开市|开盘|复牌)?(?:之日)?起|开始|复牌(?:后|之日起)?[，,]?(?:将)?(?:被)?(?:实行|实施|撤销)"
    r"|(?:开市)?复牌"
)
_NEAR = re.compile(r"起始日|起\s*[，,被实撤停]|开始|实行|实施|撤销|撤消|恢复|复牌")
_RECOUNT = re.compile(
    r"(?:之日)?起[，,]?(?:公司|本公司)?(?:股票|A股)?(?:交易)?(?:将)?(?:被)?(?:实行|实施)"
)
# Una enumeración o un punto abren otra cláusula, que ya no describe la fecha anterior.
_CLAUSE = re.compile(
    r"[（(][一二三四五六七八九十\d]{1,3}[）)]|(?<![\d.])\d{1,2}[、.](?!\d)"
    r"|[一二三四五六七八九十]{1,3}、|[；;。]"
)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _iso(value):
    """Fecha ISO desde «AAAAMMDD» o «AAAA-MM-DD»."""
    text = value.replace("-", "")
    _require(re.fullmatch(r"\d{8}", text), f"Fecha oficial no reconocible: {value!r}")
    return date(int(text[:4]), int(text[4:6]), int(text[6:])).isoformat()


def read_captures(directory):
    """Capturas con estado 200 cuyo cuerpo conserva la huella de su recibo."""
    captures = {}
    for receipt_path in sorted(Path(directory).glob("*.receipt.json")):
        receipt = json.loads(receipt_path.read_text())
        body_path = receipt_path.with_name(receipt["capture_id"] + ".body")
        if receipt.get("status") != 200 or not body_path.is_file():
            continue
        _require(body_path.stat().st_size <= MAX_CAPTURE_BYTES, "Una captura supera su límite")
        body = body_path.read_bytes()
        _require(
            hashlib.sha256(body).hexdigest() == receipt["sha256"],
            f"La captura {receipt['capture_id']} no conserva su huella",
        )
        captures[receipt["capture_id"]] = (body, receipt)
    return captures


def _xlsx_rows(data):
    """Filas de la primera hoja de un xlsx, con la biblioteca estándar."""
    space = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(data)) as book:
        shared = []
        if "xl/sharedStrings.xml" in book.namelist():
            root = ElementTree.fromstring(book.read("xl/sharedStrings.xml"))
            for item in root.findall("m:si", space):
                shared.append("".join(t.text or "" for t in item.iter(f"{{{space['m']}}}t")))
        sheet = ElementTree.fromstring(book.read("xl/worksheets/sheet1.xml"))
    for row in sheet.find("m:sheetData", space).findall("m:row", space):
        cells = {}
        for cell in row.findall("m:c", space):
            column = re.match(r"[A-Z]+", cell.get("r")).group(0)
            value = cell.find("m:v", space)
            if cell.get("t") == "s":
                cells[column] = shared[int(value.text)]
            elif cell.get("t") == "inlineStr":
                cells[column] = "".join(t.text or "" for t in cell.iter(f"{{{space['m']}}}t"))
            else:
                cells[column] = value.text if value is not None else None
        yield cells


def _listing_dates(captures):
    """Fecha oficial de admisión de cada código, de las listas de ambas bolsas."""
    listed = {}
    for capture_id in SSE_LISTS:
        for row in json.loads(captures[capture_id][0])["result"]:
            listed.setdefault(("SS", row["A_STOCK_CODE"]), _iso(row["LIST_DATE"]))
    for capture_id in SZSE_LISTS:
        rows = list(_xlsx_rows(captures[capture_id][0]))
        header = rows[0]
        code = next(k for k, v in header.items() if v in ("A股代码", "证券代码"))
        day = next(k for k, v in header.items() if v in ("A股上市日期", "上市日期"))
        for row in rows[1:]:
            if row.get(code) and row.get(day):
                listed.setdefault(("SZ", row[code]), _iso(row[day]))
    return listed


def _official_spans(captures, pattern):
    """Tramos de Shenzhen según el informe oficial de cambios de nombre, para un prefijo."""
    changes = defaultdict(list)
    for row in list(_xlsx_rows(captures[SZSE_NAMES][0]))[1:]:
        changes[row["B"]].append((_iso(row["A"]), row["D"] or "", row["E"] or ""))
    return {code: _name_spans(sorted(items), pattern) for code, items in changes.items()}


def _name_spans(changes, pattern):
    """Tramos [inicio, fin) en que el nombre cumple `pattern`. None es un extremo abierto."""
    spans, start, inside = [], None, None
    for day, before, after in changes:
        was, now = bool(pattern.match(before)), bool(pattern.match(after))
        if inside is None and was:
            start, inside = None, True
        if now and not was:
            start, inside = day, True
        elif was and not now:
            spans.append((start, day))
            start, inside = None, False
    if inside:
        spans.append((start, None))
    return spans


class _Text(HTMLParser):
    """Texto de los anuncios antiguos que CNINFO sirve en HTML, sin ejecutar nada."""

    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], 0

    def handle_starttag(self, tag, attrs):
        self.hidden += tag in ("script", "style")

    def handle_endtag(self, tag):
        self.hidden -= tag in ("script", "style")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


def announcement_text(body, cache):
    """Texto del cuerpo sin espacios: pdftotext para los PDF y el analizador HTML para el resto.

    El texto de un PDF se guarda por la huella de su contenido, de modo que una captura
    distinta nunca reutiliza un texto anterior.
    """
    if body[:4] == b"%PDF":
        target = Path(cache) / (hashlib.sha256(body).hexdigest() + ".txt")
        if not target.is_file():
            source = target.with_suffix(".pdf")
            source.write_bytes(body)
            try:
                subprocess.run(
                    [PDFTOTEXT, "-layout", "-enc", "UTF-8", str(source), str(target)],
                    check=False,
                    capture_output=True,
                    timeout=60,
                )
            finally:
                source.unlink()
        text = (
            target.read_bytes()[:MAX_TEXT_BYTES].decode("utf-8", "replace")
            if target.is_file()
            else ""
        )
    else:
        head = body[:2048].decode("ascii", "replace")
        match = re.search(r"charset\s*=\s*[\"']?([\w-]+)", head, re.IGNORECASE)
        encoding = (match.group(1) if match else "gb18030").lower()
        encoding = "gb18030" if encoding in ("gb2312", "gbk") else encoding
        parser = _Text()
        parser.feed(body.decode(encoding, "replace"))
        parser.close()
        text = "".join(parser.parts)
    return re.sub(r"\s+", "", text)


def title_kind(title):
    """Sentido del anuncio según su título, o None si no fija un cambio de estado.

    Los avisos de una posibilidad, los avances, las solicitudes y las advertencias sobre
    bonos no cambian la banda de la acción. «Continuar» o «acumular» la advertencia la
    mantiene. Retirar una advertencia e implantar otra en el mismo anuncio es un cambio
    dentro del tramo.
    """
    if re.search(
        r"债券|可转债|公司债|ST\S{0,4}债|\d{2}\S{1,4}债|申请|可能|存在被|进展|第[一二三四五六七八九十\d]+次",
        title,
    ):
        return None
    if re.search(r"撤销|撤消", title):
        rest = re.split(r"撤销|撤消", title)[-1]
        if re.search(r"(?:并|及|暨|同时|与)(?:继续)?(?:被)?(?:叠加)?(?:实施|实行)", rest):
            return "switch"
        return "revoke"
    if re.search(r"继续|叠加(?:实施|实行)", title) and re.search(r"风险警示|特别处理", title):
        return "continue"
    if re.search(r"恢复上市(?:的)?公告|恢复上市暨复牌|恢复上市首日", title) and not re.search(
        _NOT_FREE, title
    ):
        return "relisting"
    if re.search(r"法院裁定受理(?:本)?公司(?:进行)?(?:司法)?(?:破产)?重整", title):
        return "implement"
    if re.search(r"实施|实行|被", title) and re.search(r"风险警示|特别处理", title):
        return "implement"
    if title == "关于退市风险警示提示性的公告":
        return "implement"
    return None


def _candidates(body, published):
    """Fechas candidatas a efectivas, puntuadas por la expresión que las acompaña."""
    found, recounts = [], []
    for match in _DATE.finditer(body):
        try:
            day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            continue
        before = body[max(0, match.start() - 14) : match.start()]
        after = re.sub(
            r"^[（(](?:星期|周)[一二三四五六日天][）)]", "", body[match.end() : match.end() + 24]
        )
        after = _CLAUSE.split(after)[0]
        if (
            _RECOUNT.match(after)
            and re.search(r"风险警示|特别处理", after)
            and day < published - timedelta(days=45)
        ):
            recounts.append(day)
        if not EFFECTIVE_WINDOW[0] <= (day - published).days <= EFFECTIVE_WINDOW[1]:
            continue
        if "停牌" in after[:8] or re.search(r"停牌(?:的)?(?:起始)?日(?:期)?[：:为]?$", before):
            continue
        if "起始日" in before[-8:]:
            found.append((3, day))
        elif _EFFECTIVE.match(after):
            found.append((2, day))
        elif _NEAR.search(before + after):
            found.append((1, day))
    if not found:
        # Anuncios antiguos con fechas sin año, como «2月5日复牌».
        for match in re.finditer(
            r"(?<![年\d])(\d{1,2})月(\d{1,2})日(?:起)?(?:开市起)?(?:复牌|恢复交易|起撤销|起实行|起实施)",
            body,
        ):
            try:
                day = date(published.year, int(match.group(1)), int(match.group(2)))
            except ValueError:
                continue
            if EFFECTIVE_WINDOW[0] <= (day - published).days <= EFFECTIVE_WINDOW[1]:
                found.append((1, day))
    return found, sorted(set(recounts))


def read_event(row, body, sessions):
    """Cambio de estado que fija un anuncio: tipo, fecha efectiva y fechas de inicio relatadas.

    La fecha efectiva es la única de mayor puntuación. Si no hay ninguna y el anuncio dice
    que la acción para un día, el cambio se aplica en la sesión siguiente del calendario.
    """
    title = re.sub(r"<[^>]+>", "", row["announcementTitle"])
    published = datetime.fromtimestamp(row["announcementTime"] / 1000, CHINA).date()
    kind = title_kind(title)
    found, recounts = _candidates(body, published)
    top = max((score for score, _ in found), default=None)
    effective = sorted({day for score, day in found if score == top})
    if not found:
        halt = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日停牌一天", body)
        if halt:
            day = date(*(int(group) for group in halt.groups()))
            if EFFECTIVE_WINDOW[0] <= (day - published).days <= EFFECTIVE_WINDOW[1]:
                index = bisect_right(sessions, day)
                effective = [sessions[index]] if index < len(sessions) else []
    renames = [(m.group(1), m.group(2)) for m in _RENAME.finditer(body)]
    if kind == "revoke":
        # Una banda del 10 % o un nombre nuevo sin ST retiran la advertencia. Un nombre nuevo
        # con ST, o la banda que «sigue en el 5 %» sin nombre nuevo, solo la cambian. Una
        # acción sin reforma accionarial también sigue en el 5 % al perder el ST, pero su
        # nombre nuevo (S前锋) ya no lo lleva.
        leaving = [
            r for r in renames if not ST_NAME.match(r[1]) and (r[0] is None or ST_NAME.match(r[0]))
        ]
        staying = [
            r for r in renames if ST_NAME.match(r[1]) and (r[0] is None or ST_NAME.match(r[0]))
        ]
        if _TEN.search(body) or leaving:
            kind = "end"
        else:
            kind = "switch" if staying or _STILL_FIVE.search(body) else "end"
    elif kind == "implement":
        # Si el valor ya llevaba ST al publicarse, se cambia de advertencia sin abrir un tramo.
        entering = [
            r for r in renames if ST_NAME.match(r[1]) and not (r[0] and ST_NAME.match(r[0]))
        ]
        staying = [r for r in renames if ST_NAME.match(r[1]) and r[0] and ST_NAME.match(r[0])]
        already = bool(ST_NAME.match(row["secName"] or ""))
        kind = "switch" if already or staying and not entering else "start"
    elif kind == "relisting":
        # La reanudación fija la banda desde su segunda sesión: 5 % si se implanta otra
        # advertencia y 10 % si se retira la de exclusión sin sustituirla.
        if re.search(r"(?:实行|实施)其他(?:特别处理|风险警示)", body):
            kind = "relisting_st"
        elif _TEN.search(body) or re.search(r"撤销(?:公司股票)?(?:交易)?(?:的)?退市风险警示", body):
            kind = "relisting_end"
        elif _FIVE.search(body):
            kind = "relisting_st"
        else:
            kind = None
    return dict(
        id=str(row["announcementId"]),
        code=row["secCode"],
        title=title,
        published=published.isoformat(),
        kind=kind,
        effective=effective[0].isoformat() if len(effective) == 1 else None,
        recounts=[day.isoformat() for day in recounts],
    )


def no_limit_day(body, published, sessions):
    """Sesión sin límite diario que fija un anuncio de reforma accionarial o de reanudación.

    Para cada frase que declara la exención («不设涨跌幅限制», también con «跌涨» o con
    «不实行») se toma la fecha más cercana que la precede, a menos de 80 caracteres, o si no
    la primera que la sigue a menos de 40. Así se lee tanto «复牌日：2013年8月20日，当日...»
    como «首日（即2021年8月10日）不实行...», sin confundirla con otras fechas de una tabla.
    Solo vale una sesión única de la ventana del anuncio. Si ninguna frase lleva fecha, vale
    la de «自...起在...恢复上市» del mismo texto. Sin fecha única, devuelve None.
    """
    allowed = {
        day
        for day in sessions
        if EFFECTIVE_WINDOW[0] <= (day - published).days <= EFFECTIVE_WINDOW[1]
    }
    dates = [(m.start(), m.end(), _full_date(m)) for m in _DATE.finditer(body)]
    days = set()
    for phrase in _NO_LIMIT.finditer(body):
        before = [d for s, e, d in dates if e <= phrase.start() and phrase.start() - e <= 80]
        after = [d for s, e, d in dates if s >= phrase.end() and s - phrase.end() <= 40]
        chosen = before[-1] if before else after[0] if after else None
        if chosen in allowed:
            days.add(chosen)
    if not days and _NO_LIMIT.search(body):
        days = {
            _full_date(m)
            for m in re.finditer(_DATE.pattern + r"起在(?:上海|深圳)证券交易所恢复上市", body)
        } & allowed
    return min(days).isoformat() if len(days) == 1 else None


def _full_date(match):
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def _metadata(captures):
    """Filas de anuncios de todas las páginas CNINFO capturadas, sin repetir identificadores."""
    rows = {}
    for capture_id, (body, receipt) in captures.items():
        if not capture_id.startswith("cninfo-") or capture_id.startswith("cninfo-pdf-"):
            continue
        if "hisAnnouncement/query" not in receipt["url"]:
            continue
        for row in json.loads(body).get("announcements") or []:
            if row.get("secCode"):
                rows[str(row["announcementId"])] = (row, capture_id)
    return rows


def announcement_spans(events, observations, *, coverage_from=COVERAGE_FROM):
    """Tramos de un valor a partir de sus anuncios, con las contradicciones encontradas.

    Una implantación abre un tramo y una retirada lo cierra. Un cambio interno, una
    continuación o una reanudación con advertencia afirman el estado ST en su fecha y abren
    el tramo si estaba cerrado. Cuando un tramo aparece sin implantación (por una retirada o
    una afirmación), su inicio es la fecha relatada en el propio anuncio o, si la primera
    evidencia del estado es anterior a `coverage_from`, esa fecha de cobertura. Si no hay
    ninguna de las dos, es una contradicción. Cada nombre observado en CNINFO afirma el
    estado del día de publicación, salvo el mismo día de un cambio, cuando el nombre todavía
    puede ser el anterior.
    """
    order = {
        "start": 0,
        "switch": 1,
        "continue": 1,
        "relisting_st": 1,
        "end": 2,
        "relisting_end": 2,
    }
    marks = sorted({(e["effective"], order[e["kind"]], e["kind"]) for e in events})
    spans, start, problems = [], None, []

    def opened(day, previous):
        """Inicio de un tramo que aparece sin implantación, o None si no consta."""
        recounted = [
            r for e in events if e["effective"] == day for r in e["recounts"] if previous < r < day
        ]
        if recounted:
            return max(recounted)
        evidence = [d for d, _, k in marks if order[k] == 1 and previous < d <= day]
        evidence += [d for d, st in observations if st and previous < d <= day]
        return coverage_from if evidence and min(evidence) < coverage_from else None

    for day, rank, kind in marks:
        previous = spans[-1][1] if spans else ""
        if rank == 0:
            start = day if start is None else start
        elif rank == 1 and start is None:
            start = opened(day, previous)
            if start is None and day >= coverage_from:
                problems.append(dict(problem="asserted_without_start", day=day))
            start = start or day
        elif rank == 2 and start is not None:
            spans.append([start, day])
            start = None
        elif kind == "end":
            begun = opened(day, previous)
            if begun is not None:
                spans.append([begun, day])
            elif day > coverage_from:
                problems.append(dict(problem="end_without_start", day=day))
    if start is not None:
        spans.append([start, None])
    changes = {day for day, rank, _ in marks if rank != 1}
    for day, st in observations:
        inside = any(a <= day and (b is None or day < b) for a, b in spans)
        if day >= coverage_from and day not in changes and inside != st:
            problems.append(dict(problem="name_disagrees", day=day, special_treatment=st))
    return spans, problems


def _clip(spans, coverage_from=COVERAGE_FROM, cutoff=CUTOFF):
    """Tramos visibles desde la cobertura hasta el corte. Un final posterior queda abierto."""
    result = []
    for start, end in spans:
        if end is not None and end <= coverage_from or start is not None and start > cutoff:
            continue
        start = coverage_from if start is None or start < coverage_from else start
        result.append([start, None if end is None or end > cutoff else end])
    return result


def _daily(spans, sessions):
    return {day for day in sessions if any(a <= day and (b is None or day < b) for a, b in spans)}


def _reform_spans(free_days, observations, *, coverage_from=COVERAGE_FROM):
    """Tramo sin reforma accionarial de Shanghái y sus contradicciones con los nombres.

    Desde octubre de 2006 toda acción sin reforma llevaba el prefijo S, y la reforma no se
    deshace. Así que un valor reformado después de `coverage_from` estaba en S desde esa fecha
    hasta su primer día sin límite. Ese día lo fija el anuncio de la reforma.
    """
    reformed = [day for day, kind in free_days if kind == "reform" and day >= coverage_from]
    spans = [[coverage_from, min(reformed)]] if reformed else []
    problems = []
    for day, pending in observations:
        inside = any(a <= day < b for a, b in spans)
        if day >= coverage_from and inside != pending and day not in reformed:
            problems.append(dict(problem="share_reform_name_disagrees", day=day, pending=pending))
    return spans, problems


def price_band_check(edition, assets, china, sessions, tolerance=0.02):
    """Contrastar la tabla con los precios sin ajustar: cierres fuera de la banda del día.

    Se comparan pares de sesiones consecutivas del calendario, ambas verificadas y con
    negociación, sin acción corporativa y desde la cobertura. Una suspensión entre dos filas
    acumularía la variación de varias sesiones y una fila sin volumen no tiene cierre
    negociado, así que esos pares no se cuentan. La tolerancia de dos céntimos absorbe el
    redondeo de la reconstrucción. Un cierre fuera de la banda declarada señala un error de
    la tabla o de los precios, y se lista.
    """
    import pyarrow.parquet as parquet

    from mars_titan.simulation.market_rules import beijing_day, china_a_share_instrument

    position = {day.isoformat(): index for index, day in enumerate(sessions)}
    checked, outside = defaultdict(int), []
    for key in assets:
        limits = china_a_share_instrument(key, china[key]).price_limits
        prices = parquet.read_table(
            edition / "assets" / key / "prices.parquet",
            columns=["session", "close", "volume", "verified"],
        ).to_pylist()
        events = set(
            parquet.read_table(edition / "assets" / key / "events.parquet", columns=["session"])
            .column(0)
            .to_pylist()
        )
        for previous, row in zip(prices, prices[1:], strict=False):
            day = row["session"]
            if day < COVERAGE_FROM or day > CUTOFF or day in events:
                continue
            if position.get(day, -1) != position.get(previous["session"], -2) + 1:
                continue
            if not (previous["verified"] and row["verified"]) or not previous["close"] > 0:
                continue
            if not (previous["volume"] > 0 and row["volume"] > 0):
                continue
            at = beijing_day(*date.fromisoformat(day).timetuple()[:3])
            period = next((p for p in limits if p.start <= at < p.end), None)
            band = None if period is None else period.band
            checked[str(band)] += 1
            if (
                band is not None
                and abs(row["close"] - previous["close"]) > band * previous["close"] + tolerance
            ):
                outside.append([key, day, band, round(row["close"] / previous["close"] - 1, 4)])
    by_band = defaultdict(int)
    for item in outside:
        by_band[str(item[2])] += 1
    # Los cierres fuera de la banda reducida contrastan los tramos. Los demás, una muestra.
    reduced = [item for item in outside if item[2] == SPECIAL_BAND]
    return dict(
        sessions_by_band=dict(checked),
        outside_by_band=dict(by_band),
        outside_reduced_band=reduced,
        outside_other_sample=[item for item in outside if item[2] != SPECIAL_BAND][:100],
    )


def build_listing_status(captures_dir, edition, output, *, texts):
    """Escribir `listing-status.json` y su informe. Devuelve el informe."""
    from mars_titan.simulation.market_rules import limit_free_until

    captures_dir, edition, output, texts = (Path(p) for p in (captures_dir, edition, output, texts))
    outside_source(captures_dir, output)
    texts.mkdir(parents=True, exist_ok=True)
    captures = read_captures(captures_dir)
    manifest = json.loads((edition / "manifest.json").read_text())
    assets = sorted(key for key in manifest["receipts"] if key.startswith("CN/"))
    codes = {key[3:9] for key in assets}
    sessions = MarketClock("CN", "2000-01-01", "2024-02-29").days
    cutoff = date.fromisoformat(CUTOFF)
    covered = [d.isoformat() for d in sessions if COVERAGE_FROM <= d.isoformat() <= CUTOFF]
    listed = _listing_dates(captures)
    official = _official_spans(captures, ST_NAME)
    official_reform = _official_spans(captures, S_NAME)
    events, free_days = defaultdict(list), defaultdict(set)
    observations, reform_names = defaultdict(list), defaultdict(list)
    used = set(SSE_LISTS + SZSE_LISTS + (SZSE_NAMES,))
    pending, undated = [], []
    for aid, (row, page) in sorted(_metadata(captures).items()):
        published = datetime.fromtimestamp(row["announcementTime"] / 1000, CHINA).date()
        if published > cutoff + timedelta(days=EFFECTIVE_WINDOW[1]):
            continue
        code, name = row["secCode"], row["secName"] or ""
        if code not in codes:
            continue
        observations[code].append((published.isoformat(), bool(ST_NAME.match(name))))
        reform_names[code].append((published.isoformat(), bool(S_NAME.match(name))))
        used.add(page)
        title = re.sub(r"<[^>]+>", "", row["announcementTitle"])
        frees = bool(_FREE_TITLE.search(title)) and not re.search(_NOT_FREE, title)
        if title_kind(title) is None and not frees:
            continue
        capture = captures.get(f"cninfo-pdf-{aid}")
        if capture is None:
            pending.append(dict(id=aid, code=code, title=title, published=str(published)))
            continue
        used.add(f"cninfo-pdf-{aid}")
        body = announcement_text(capture[0], texts)
        event = read_event(row, body, sessions)
        if event["kind"] and event["effective"]:
            events[code].append(event)
        if frees:
            day = no_limit_day(body, published, sessions)
            if day is None:
                undated.append(dict(id=aid, code=code, title=title, published=str(published)))
            else:
                free_days[code].add((day, "reform" if "股权分置" in title else "relisting"))
    # Concordancia del extractor con el informe oficial de Shenzhen, día a día.
    agreement = dict(both=0, official_only=0, announced_only=0, differing_codes=[])
    reform_check = []
    for key in assets:
        code, exchange = key[3:9], key[10:]
        if exchange != "SZ":
            continue
        truth = _daily(_clip(official.get(code, [])), covered)
        found, _ = announcement_spans(events[code], sorted(observations[code]))
        found = _daily(_clip(found), covered)
        agreement["both"] += len(truth & found)
        agreement["official_only"] += len(truth - found)
        agreement["announced_only"] += len(found - truth)
        if truth != found:
            agreement["differing_codes"].append(key)
        for _, end in official_reform.get(code, []):
            if end is not None and end >= COVERAGE_FROM:
                announced = sorted(day for day, kind in free_days[code] if kind == "reform")
                reform_check.append(
                    dict(asset=key, official_end=end, announced_free_days=announced)
                )
    china, problems = {}, []
    for key in assets:
        code, exchange = key[3:9], key[10:]
        _require((exchange, code) in listed, f"{key} no figura en las listas oficiales de admisión")
        if exchange == "SZ":
            # Un primer tramo sin inicio ya tenía ese nombre antes del primer cambio registrado.
            spans, reform = official.get(code, []), official_reform.get(code, [])
        else:
            spans, found = announcement_spans(events[code], sorted(observations[code]))
            reform, more = _reform_spans(sorted(free_days[code]), sorted(reform_names[code]))
            problems += [dict(asset=key, **problem) for problem in found + more]
        on = listed[(exchange, code)]
        china[key] = dict(
            listed_on=on,
            limit_free_until=limit_free_until(key, on) if on <= CUTOFF else None,
            special_treatment=_clip(spans),
            share_reform_pending=_clip(reform),
            limit_free_days=sorted(
                {day for day, _ in free_days[code] if COVERAGE_FROM <= day <= CUTOFF}
            ),
        )

    def dated_nearby(item):
        """Un aviso sin fecha repite el día que fija otro anuncio fechado del mismo valor."""
        published = date.fromisoformat(item["published"])
        return any(
            abs((date.fromisoformat(day) - published).days) <= EFFECTIVE_WINDOW[1]
            for day, _ in free_days[item["code"]]
        )

    undated = [item for item in undated if not dated_nearby(item)]
    late = [p for p in pending + undated if p["published"] >= COVERAGE_FROM]
    # Un anuncio sin texto suele causar las contradicciones, así que se informa antes.
    _require(not late, f"Anuncios de estado sin texto o sin fecha desde la cobertura: {late}")
    _require(not problems, f"Contradicciones en los tramos de Shanghái: {problems}")
    table = dict(
        kind=TABLE_KIND,
        schema_version=1,
        cutoff=CUTOFF,
        sources={
            capture_id: dict(
                sha256=captures[capture_id][1]["sha256"], url=captures[capture_id][1]["url"]
            )
            for capture_id in sorted(used)
        },
        china=china,
        exits={},
    )
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "listing-status.json", table)
    _, table_sha256 = read_listing_status(output / "listing-status.json")
    version = subprocess.run([PDFTOTEXT, "-v"], capture_output=True, text=True, check=False)
    report = dict(
        kind="listing_status_build",
        table_sha256=table_sha256,
        edition_id=manifest["edition_id"],
        built_at_utc=datetime.now(UTC).isoformat(),
        coverage_from=COVERAGE_FROM,
        cutoff=CUTOFF,
        pdftotext=(version.stderr or version.stdout).splitlines()[0],
        assets=len(china),
        special_treatment_spans={
            exchange: sum(
                len(v["special_treatment"]) for k, v in china.items() if k.endswith(exchange)
            )
            for exchange in (".SS", ".SZ")
        },
        share_reform_pending=sorted(k for k, v in china.items() if v["share_reform_pending"]),
        limit_free_days={k: v["limit_free_days"] for k, v in china.items() if v["limit_free_days"]},
        limit_free_assets=sorted(k for k, v in china.items() if v["limit_free_until"] is not None),
        sources=len(table["sources"]),
        szse_extractor_agreement=agreement,
        szse_reform_days=reform_check,
        pending_before_coverage=len(pending) + len(undated) - len(late),
        events=sum(len(v) for v in events.values()),
        price_band_check=price_band_check(edition, assets, china, sessions),
    )
    atomic_json(output / "listing-status-report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--captures", required=True, help="Directorio de capturas con recibos")
    parser.add_argument("--edition", required=True, help="Edición sin ajustar con su manifiesto")
    parser.add_argument("--texts", required=True, help="Caché de textos extraídos de los PDF")
    parser.add_argument("--output", required=True, help="Directorio de la tabla y su informe")
    args = parser.parse_args(argv)
    report = build_listing_status(args.captures, args.edition, args.output, texts=args.texts)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
