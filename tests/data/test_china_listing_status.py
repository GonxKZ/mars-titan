"""Construcción de la tabla del estado de cotización chino con capturas sintéticas.

Los textos imitan la redacción de los anuncios reales de CNINFO, pero son fixtures: no
proceden de ninguna captura. Las capturas del caso completo se escriben con sus recibos y
huellas, como las guarda el capturador.
"""

import hashlib
import json
import zipfile
from datetime import date
from io import BytesIO

import pytest

from mars_titan.data import china_listing_status as status
from mars_titan.simulation.listing_status import read_listing_status
from mars_titan.simulation.market_rules import china_a_share_instrument, xshg_sessions
from tests.simulation.unadjusted_edition_fixture import Asset, write_edition

SESSIONS = list(xshg_sessions())


def row(title, published, *, name="浦发银行", code="600000", aid="1"):
    """Fila de metadatos de CNINFO publicada a mediodía de Pekín."""
    moment = date.fromisoformat(published).toordinal() - date(1970, 1, 1).toordinal()
    return dict(
        announcementId=aid,
        secCode=code,
        secName=name,
        announcementTitle=title,
        announcementTime=(moment * 86_400 + 4 * 3_600) * 1000,
        adjunctUrl=f"finalpage/{published}/{aid}.PDF",
    )


def event(title, published, body, *, name="浦发银行"):
    return status.read_event(row(title, published, name=name), body, SESSIONS)


@pytest.mark.parametrize(
    ("title", "kind"),
    [
        ("关于公司股票实施退市风险警示的公告", "implement"),
        ("关于公司股票可能被实施退市风险警示的第二次提示性公告", None),
        ("关于申请撤销公司股票退市风险警示的公告", None),
        ("关于2011年公司债券（11天威债）实施风险警示的公告", None),
        ("关于ST天威债撤销风险警示公告", None),
        ("关于撤销公司股票退市风险警示的公告", "revoke"),
        ("关于撤销对公司股票交易实施的其他特别处理的提示性公告", "revoke"),
        ("关于撤销公司股票退市风险警示及实施其他风险警示的公告", "switch"),
        ("关于撤销相关风险警示暨继续被叠加实施退市风险警示和其他风险警示的公告", "switch"),
        ("关于公司股票被继续实施退市风险警示的公告", "continue"),
        ("关于法院裁定受理公司重整暨股票被叠加实施退市风险警示的公告", "continue"),
        ("关于公司股票被实施退市风险警示叠加其他风险警示暨临时停牌的公告", "implement"),
        ("关于法院许可公司在重整期间继续营业及自行管理财产和营业事务的公告", None),
        ("关于法院裁定受理公司重整的公告", "implement"),
        ("关于法院裁定受理控股股东重整的公告", None),
        ("关于股票恢复上市的公告", "relisting"),
        ("关于AC结合疫苗恢复上市的公告", None),
    ],
)
def test_titles_separate_changes_from_warnings_bonds_and_homonyms(title, kind):
    assert status.title_kind(title) == kind


def test_the_start_date_is_the_declared_start_and_not_the_suspension_day():
    body = (
        "（四）实施退市风险警示的起始日：2013年3月29日（星期五）"
        "（五）停牌日：2013年3月28日公司股票停牌一天。"
    )
    found = event("关于被实施退市风险警示的公告", "2013-03-28", body)
    assert (found["kind"], found["effective"]) == ("start", "2013-03-29")


def test_a_date_typo_falls_back_to_the_session_after_the_one_day_halt():
    # El anuncio real decía 2010 por 2011. El 2 de mayo de 2011 era festivo.
    body = "公司股票于2011年4月29日停牌一天，并将于2010年5月3日起撤销股票交易退市风险警示，涨跌幅限制恢复为10%。"
    found = event("关于股票撤销退市风险警示及其他特别处理的公告", "2011-04-29", body)
    assert (found["kind"], found["effective"]) == ("end", "2011-05-03")


def test_a_revocation_ends_or_switches_by_the_new_band_and_the_new_name():
    ended = event(
        "关于撤销股票交易其他特别处理的公告",
        "2011-10-25",
        "公司股票自2011年10月26日起恢复交易，证券简称由“*ST汇通”变更为“ST汇通”，"
        "公司股票交易日涨跌幅限制变为10%。",
        name="ST汇通",
    )
    switched = event(
        "关于撤销退市风险警示的公告",
        "2007-04-26",
        "自2007年4月30日起，公司股票简称变更为“ST一投”，股票涨跌幅限制仍为5%。",
        name="*ST一投",
    )
    # Pierde el ST pero sigue sin reforma accionarial, así que la banda sigue en el 5 %.
    reform = event(
        "关于撤销股票退市风险警示的公告",
        "2013-04-03",
        "公司股票2013年4月8日复牌，证券简称由“S*ST前锋”变更为“S前锋”，日涨跌幅限制仍为5%。",
        name="S*ST前锋",
    )
    assert [e["kind"] for e in (ended, switched, reform)] == ["end", "switch", "end"]
    assert [e["effective"] for e in (ended, switched, reform)] == [
        "2011-10-26",
        "2007-04-30",
        "2013-04-08",
    ]


def test_an_implementation_on_a_stock_already_under_warning_is_a_switch():
    body = "公司股票将于2010年4月1日起实施退市风险警示，实施退市风险警示后，股票简称：*ST筑信。"
    found = event("关于公司股票实行退市风险警示的公告", "2010-03-31", body, name="ST筑信")
    assert (found["kind"], found["effective"]) == ("switch", "2010-04-01")


def test_a_relisting_keeps_or_drops_the_warning_from_its_second_session():
    other = "公司股票自2009年11月13日起恢复上市，撤销退市风险警示，实行其他特别处理，以后每个交易日涨跌幅限制为5%。"
    clean = "公司股票自2011年9月29日起恢复上市，并撤销退市风险警示，自第二个交易日起涨跌幅限制为10%。"
    assert event("关于股票恢复上市的公告", "2009-11-09", other)["kind"] == "relisting_st"
    assert event("关于股票恢复上市的公告", "2011-09-23", clean)["kind"] == "relisting_end"


@pytest.mark.parametrize(
    ("published", "body", "expected"),
    [
        (
            "2013-08-13",
            "2、公司A股股票复牌日及对价股票上市日：2013年8月20日。当日股价不计算除权参考价、"
            "不设涨跌幅度限制，不纳入指数计算。",
            "2013-08-20",
        ),
        (
            "2021-08-09",
            "公司股票恢复上市首日（即2021年8月10日）不实行价格涨跌幅限制，"
            "自恢复上市次一交易日（即2021年8月11日）起涨跌幅限制为5%。",
            "2021-08-10",
        ),
        (
            "2018-09-25",
            "公司股票将于2018年9月27日复牌，当日公司股票不设跌涨幅限制。",
            "2018-09-27",
        ),
        (
            "2010-02-10",
            "32010年2月12日恢复交易该日公司股票不计算除权参考价、不设涨跌幅限制、不纳入指数计算"
            "公司股票开始设涨跌幅限制，以前一交易日为基42010年2月22日正常交易期纳入指数计算",
            "2010-02-12",
        ),
        (
            "2011-08-23",
            "本公司A股股票自2011年8月29日起在上海证券交易所恢复上市。恢复上市的第一个交易日不设涨跌幅限制。",
            "2011-08-29",
        ),
        ("2015-08-10", "待公告复牌后，首个交易日不设涨跌幅限制。", None),
    ],
)
def test_the_no_limit_day_is_the_date_next_to_its_declaration(published, body, expected):
    assert status.no_limit_day(body, date.fromisoformat(published), SESSIONS) == expected


def mark(kind, effective, recounts=()):
    return dict(kind=kind, effective=effective, recounts=list(recounts))


def test_spans_open_close_and_switch_without_reopening():
    events = [
        mark("start", "2018-04-24"),
        mark("switch", "2019-03-28"),
        mark("continue", "2021-04-30"),
        mark("end", "2022-05-06"),
    ]
    names = [("2018-04-23", False), ("2019-03-27", True), ("2022-05-05", True), ("2022-06-01", False)]
    assert status.announcement_spans(events, names) == ([["2018-04-24", "2022-05-06"]], [])


def test_a_span_without_its_implementation_needs_a_recount_or_earlier_evidence():
    recounted = status.announcement_spans(
        [mark("end", "2014-04-03", recounts=["2013-03-29"])], [("2014-03-01", True)]
    )
    assert recounted == ([["2013-03-29", "2014-04-03"]], [])
    # La primera evidencia es anterior a la cobertura: el tramo empieza en ella.
    earlier = status.announcement_spans(
        [mark("switch", "2008-04-25"), mark("end", "2012-08-22")], [("2009-07-02", True)]
    )
    assert earlier == ([["2010-01-01", "2012-08-22"]], [])
    # Sin relato ni evidencia anterior, el inicio no consta y es una contradicción.
    unknown = status.announcement_spans([mark("end", "2021-04-30")], [("2021-01-11", True)])
    # La retirada sin inicio deja además fuera de tramo el nombre ST observado antes.
    assert unknown[1] == [
        dict(problem="end_without_start", day="2021-04-30"),
        dict(problem="name_disagrees", day="2021-01-11", special_treatment=True),
    ]
    asserted = status.announcement_spans([mark("switch", "2021-01-19")], [])
    assert asserted[1] == [dict(problem="asserted_without_start", day="2021-01-19")]


def test_a_name_that_contradicts_the_spans_is_reported():
    events = [mark("start", "2018-04-24"), mark("end", "2019-04-02")]
    _, problems = status.announcement_spans(events, [("2018-06-01", False), ("2019-06-01", True)])
    assert problems == [
        dict(problem="name_disagrees", day="2018-06-01", special_treatment=False),
        dict(problem="name_disagrees", day="2019-06-01", special_treatment=True),
    ]
    # El nombre del mismo día de un cambio todavía puede ser el anterior.
    assert status.announcement_spans(events, [("2018-04-24", False)])[1] == []


def test_official_name_changes_give_warning_and_share_reform_spans():
    changes = [
        ("2009-04-21", "S ST集琦", "S*ST集琦"),
        ("2010-08-06", "S*ST集琦", "SST集琦"),
        ("2011-08-09", "SST集琦", "国海证券"),
        ("2017-05-03", "国海证券", "*ST国海"),
    ]
    warning = status._name_spans(changes, status.ST_NAME)
    reform = status._name_spans(changes, status.S_NAME)
    assert warning == [(None, "2011-08-09"), ("2017-05-03", None)]
    assert reform == [(None, "2011-08-09")]
    assert status._clip(warning) == [["2010-01-01", "2011-08-09"], ["2017-05-03", None]]
    for name, pending in (("S前锋", True), ("ST前锋", False), ("SST前锋", True), ("*ST前锋", False)):
        assert bool(status.S_NAME.match(name)) is pending


def test_share_reform_spans_end_on_the_first_day_without_limit():
    spans, problems = status._reform_spans(
        [("2013-08-20", "reform")], [("2012-03-01", True), ("2013-09-01", False)]
    )
    assert (spans, problems) == ([["2010-01-01", "2013-08-20"]], [])
    _, problems = status._reform_spans([], [("2012-03-01", True)])
    assert problems == [dict(problem="share_reform_name_disagrees", day="2012-03-01", pending=True)]


# Caso completo con capturas sintéticas.


def xlsx(rows):
    """Libro mínimo con una hoja de cadenas en línea."""
    columns = "ABCDEFGHIJ"
    cells = "".join(
        f'<row r="{r + 1}">'
        + "".join(
            f'<c r="{columns[c]}{r + 1}" t="inlineStr"><is><t>{value}</t></is></c>'
            for c, value in enumerate(values)
        )
        + "</row>"
        for r, values in enumerate(rows)
    )
    sheet = (
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{cells}</sheetData></worksheet>"
    )
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as book:
        book.writestr("xl/worksheets/sheet1.xml", sheet)
    return buffer.getvalue()


def capture(folder, capture_id, body, url="https://example.invalid/fixture"):
    body = body if isinstance(body, bytes) else body.encode()
    (folder / f"{capture_id}.body").write_bytes(body)
    receipt = dict(capture_id=capture_id, status=200, url=url, sha256=hashlib.sha256(body).hexdigest())
    (folder / f"{capture_id}.receipt.json").write_text(json.dumps(receipt))


def html(text):
    return f'<html><head><meta charset="utf-8"></head><body><p>{text}</p></body></html>'


@pytest.fixture
def sources(tmp_path):
    edition = tmp_path / "edition"
    write_edition(edition, {"CN": [Asset("600000.SS", base=10.0), Asset("000001.SZ", base=12.0)]})
    folder = tmp_path / "captures"
    folder.mkdir()
    sse = [dict(A_STOCK_CODE="600000", LIST_DATE="19991110")]
    capture(folder, "sse-gp-l-common-type1", json.dumps(dict(result=sse)))
    capture(folder, "sse-gp-l-common-type8", json.dumps(dict(result=[])))
    capture(folder, "sse-terminated-main", json.dumps(dict(result=[])))
    capture(
        folder,
        "szse-a-share-list-1110-xlsx",
        xlsx([["板块", "A股代码", "A股上市日期"], ["主板", "000001", "1991-04-03"]]),
    )
    for tab in ("tab1", "tab2"):
        capture(
            folder,
            f"szse-suspended-terminated-1793-{tab}-xlsx",
            xlsx([["证券代码", "证券简称", "上市日期"]]),
        )
    capture(
        folder,
        "szse-changename-tab2-xlsx",
        xlsx(
            [
                ["变更日期", "证券代码", "证券简称", "变更前简称", "变更后简称"],
                ["2023-05-05", "000001", "ST平安", "平安银行", "ST平安"],
                ["2023-09-01", "000001", "平安银行", "ST平安", "平安银行"],
            ]
        ),
    )
    rows = [
        row("关于公司股票实施退市风险警示的公告", "2023-03-30", aid="11"),
        row("关于撤销公司股票退市风险警示的公告", "2023-08-10", name="*ST浦发", aid="12"),
        row("2023年半年度报告", "2023-08-30", name="浦发银行", aid="13"),
    ]
    page = dict(announcements=rows)
    url = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
    capture(folder, "cninfo-sse-title-fixture-2023-page1", json.dumps(page), url=url)
    capture(
        folder,
        "cninfo-pdf-11",
        html("（四）实施退市风险警示的起始日：2023年4月3日（星期一），股票简称由“浦发银行”变更为“*ST浦发”。"),
    )
    capture(
        folder,
        "cninfo-pdf-12",
        html(
            "公司股票自2023年8月14日起撤销退市风险警示，股票简称由“*ST浦发”变更为“浦发银行”，"
            "日涨跌幅限制由5%变更为10%。"
        ),
    )
    return folder, edition, tmp_path


def build(sources):
    folder, edition, root = sources
    return status.build_listing_status(folder, edition, root / "table", texts=root / "texts")


def test_the_table_joins_official_lists_names_and_announcements(sources):
    report = build(sources)
    table, digest = read_listing_status(sources[2] / "table" / "listing-status.json")
    assert digest == report["table_sha256"]
    assert table["china"]["CN/600000.SS"] == dict(
        listed_on="1999-11-10",
        limit_free_until=None,
        special_treatment=[["2023-04-03", "2023-08-14"]],
        share_reform_pending=[],
        limit_free_days=[],
    )
    assert table["china"]["CN/000001.SZ"]["special_treatment"] == [["2023-05-05", "2023-09-01"]]
    assert set(table["sources"]) >= {"cninfo-pdf-11", "cninfo-pdf-12", "szse-changename-tab2-xlsx"}
    assert table["exits"] == {}
    # Las bandas derivadas cambian al 5 % dentro del tramo.
    rules = china_a_share_instrument("CN/600000.SS", table["china"]["CN/600000.SS"])
    assert sorted({period.band for period in rules.price_limits}) == [0.05, 0.1]
    assert report["price_band_check"]["sessions_by_band"]["0.05"] > 0


def test_a_tampered_capture_or_a_contradiction_stops_the_build(sources):
    folder, _, _ = sources
    body = folder / "cninfo-pdf-12.body"
    original = body.read_bytes()
    body.write_bytes(original + b" ")
    with pytest.raises(ValueError, match="huella"):
        build(sources)
    body.write_bytes(original)
    # Un nombre ST observado fuera de los tramos contradice los anuncios.
    page = folder / "cninfo-sse-title-fixture-2023-page1.body"
    data = json.loads(page.read_text())
    data["announcements"][2]["secName"] = "*ST浦发"
    capture(folder, "cninfo-sse-title-fixture-2023-page1", json.dumps(data), url="https://www.cninfo.com.cn/new/hisAnnouncement/query")
    with pytest.raises(ValueError, match="Contradicciones"):
        build(sources)


def test_a_state_announcement_without_its_text_stops_the_build(sources):
    folder, _, _ = sources
    (folder / "cninfo-pdf-12.receipt.json").unlink()
    with pytest.raises(ValueError, match="sin texto"):
        build(sources)
