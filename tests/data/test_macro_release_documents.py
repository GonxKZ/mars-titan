"""Lecturas de comunicados fechados sin mezclar índices, periodos o acumulados."""

import importlib

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.macro_release_documents")
    except ModuleNotFoundError:
        pytest.fail("Falta el lector de comunicados macro oficiales")


def document(title, table):
    return f"""<html><h1 class="con_titles">{title}</h1>
    <div class="info"><span>National Bureau of Statistics of China</span>
    <span>2023-02-11 09:30</span></div><div class="TRS_Editor">{table}</div></html>""".encode()


URL = "https://www.stats.gov.cn/english/PressRelease/202302/t20230213_1902713.html"


def test_cpi_uses_year_on_year_not_monthly_or_annual_average():
    table = """<table><tr><td></td><td>M/M (%)</td><td>Y/Y (%)</td><td>Annual (%)</td></tr>
    <tr><td>Consumer Prices</td><td>0.5</td><td>-0.2</td><td>2.4</td></tr></table>"""
    result = module().extract_nbs_release(document("Consumer Prices for January 2023", table), URL)
    assert len(result["observations"]) == 1
    row = result["observations"][0]
    assert row["indicator_id"] == "cn_cpi_yoy_published"
    assert row["period_start"] == "2023-01-01"
    assert row["value"] == -0.2
    assert row["realtime_start"] == "2023-02-13"
    assert row["declared_publication_date"] == "2023-02-11"


def test_cpi_accepts_published_tables_entirely_inside_thead():
    table = """<table><thead><tr><th>Item</th><th>M/M (%)</th><th>Y/Y (%)</th></tr>
    <tr><th>Consumer Prices</th><th>-0.2</th><th>0.0</th></tr></thead></table>"""
    result = module().extract_nbs_release(document("Consumer Prices for January 2023", table), URL)
    assert result["observations"][0]["value"] == 0.0


def test_industry_monthly_column_is_separate_from_year_to_date():
    table = """<table><tr><td rowspan="2"></td><td colspan="2">January</td>
    <td colspan="2">Jan-Jan</td></tr><tr><td>Value</td><td>Growth rate Y/Y (%)</td>
    <td>Value</td><td>Growth rate Y/Y (%)</td></tr>
    <tr><td>Value-added of Industries Above the Designated Size</td>
    <td>...</td><td>4.5</td><td>...</td><td>2.2</td></tr></table>"""
    result = module().extract_nbs_release(
        document("Industrial Production Operation in January 2023", table), URL
    )
    assert result["observations"][0]["value"] == 4.5


def test_combined_january_february_is_explicitly_excluded_from_monthly_series():
    table = """<table><tr><td></td><td>Jan-Feb</td><td>Growth rate Y/Y (%)</td></tr>
    <tr><td>Value-added of Industry Above Designated Size</td>
    <td>...</td><td>2.4</td></tr></table>"""
    result = module().extract_nbs_release(
        document("Industrial Production Operation from January to February 2023", table), URL
    )
    assert result["observations"] == []
    assert result["exclusion"] == "nonmonthly_reference_period"


def test_pmi_selects_manufacturing_and_keeps_versions_of_prior_months():
    table = """<p>China's Manufacturing PMI (Seasonally Adjusted)</p><table>
    <tr><td></td><td>PMI</td><td>Production Index</td></tr>
    <tr><td>2022-December</td><td>47.0</td><td>44.0</td></tr>
    <tr><td>2023-January</td><td>50.1</td><td>49.8</td></tr></table>
    <table><tr><td></td><td>Business Activity Index</td></tr>
    <tr><td>2023-January</td><td>54.4</td></tr></table>"""
    result = module().extract_nbs_release(
        document("Purchasing Managers Index for January 2023", table), URL
    )
    rows = result["observations"]
    assert [(r["period_start"], r["value"]) for r in rows] == [
        ("2022-12-01", 47.0),
        ("2023-01-01", 50.1),
    ]
    assert all(r["realtime_start"] == "2023-02-13" for r in rows)


def test_untrusted_host_and_oversized_table_spans_are_rejected():
    page = document(
        "Consumer Prices for January 2023",
        '<table><tr><td colspan="999999">Consumer Prices</td></tr></table>',
    )
    with pytest.raises(ValueError):
        module().extract_nbs_release(page, URL)
    with pytest.raises(ValueError):
        module().extract_nbs_release(page, "https://example.com/report.html")


PBC_URL = "https://www.pbc.gov.cn/diaochatongjisi/116219/116225/abcdef1234/index.html"
GD_URL = "https://www.gdjr.gov.cn/gdjr/zwgk/zdly/sjfb/tjsj/content/post_19054.html"


def financial_document(title, body, *, mirror=False):
    if mirror:
        return f"""<html><div class="title_content">{title}</div>
        <li class="c_time">发布时间：2023-12-14 15:28</li>
        <span id="ly">中国人民银行</span><div id="zoom">{body}</div></html>""".encode()
    return f"""<html><meta name="PubDate" content="2023-12-13">
    <h2>{title}</h2><span id="shijian">2023-12-13 17:01:00</span>
    <div id="zoom">{body}</div></html>""".encode()


@pytest.mark.parametrize("mirror", [False, True])
def test_m2_is_stock_in_billions_and_reprint_uses_its_own_date(mirror):
    result = module().extract_pboc_release(
        financial_document(
            "2023年11月金融统计数据报告",
            "11月末，广义货币（M2）余额291.2万亿元，同比增长10%。狭义货币（M1）余额67.59万亿元。",
            mirror=mirror,
        ),
        GD_URL if mirror else PBC_URL,
    )
    row = result["observations"][0]
    assert row["indicator_id"] == "cn_m2_stock"
    assert row["value"] == 291200.0
    assert row["period_start"] == "2023-11-01"
    assert row["native_unit"] == "billion_CNY_normalized_from_release"
    assert row["source_unit"] == "万亿元"
    assert row["realtime_start"] == ("2023-12-14" if mirror else "2023-12-13")


@pytest.mark.parametrize("amount,expected", [("5282亿元", 528.2), ("2.45万亿元", 2450.0)])
def test_tsf_flow_uses_monthly_value_not_cumulative_or_loans(amount, expected):
    result = module().extract_pboc_release(
        financial_document(
            "2023年11月社会融资规模增量统计数据报告",
            f"2023年前十一个月社会融资规模增量累计为33.65万亿元。"
            f"11月份社会融资规模增量为{amount}。对实体经济发放的人民币贷款增加1.11万亿元。",
        ),
        PBC_URL,
    )
    row = result["observations"][0]
    assert row["indicator_id"] == "cn_tsf_flow"
    assert row["value"] == expected
    assert row["native_unit"] == "billion_CNY_per_month_normalized_from_release"


def test_annual_tsf_stock_is_december_but_annual_flow_is_not_monthly():
    result = module().extract_pboc_release(
        financial_document(
            "2022年社会融资规模存量统计数据报告",
            "初步统计，2022年末社会融资规模存量为344.21万亿元。",
        ),
        PBC_URL,
    )
    assert result["observations"][0]["period_start"] == "2022-12-01"
    assert result["observations"][0]["value"] == 344210.0
    result = module().extract_pboc_release(
        financial_document(
            "2022年社会融资规模增量统计数据报告", "2022年全年社会融资规模增量累计为32.01万亿元。"
        ),
        PBC_URL,
    )
    assert result["observations"] == []
    assert result["exclusion"] == "monthly_flow_not_published"


def test_quarter_report_requires_explicit_monthly_flow():
    result = module().extract_pboc_release(
        financial_document(
            "2023年前三季度社会融资规模增量统计数据报告",
            "前三季度社会融资规模增量累计为29.33万亿元。9月份，社会融资规模增量为4.12万亿元。",
        ),
        PBC_URL,
    )
    assert result["observations"][0]["period_start"] == "2023-09-01"
    assert result["observations"][0]["value"] == 4120.0


@pytest.mark.parametrize("replacement", ["NaN", "Inf", "0", "-2.5", "1e300"])
def test_monetary_stock_rejects_nonpositive_or_unrecognised_amounts(replacement):
    with pytest.raises(ValueError):
        module().extract_pboc_release(
            financial_document(
                "2023年11月金融统计数据报告", f"11月末，广义货币（M2）余额{replacement}万亿元。"
            ),
            PBC_URL,
        )


def test_financial_release_rejects_ambiguous_period_provenance_or_correction():
    content = financial_document(
        "2023年11月金融统计数据报告", "10月末，广义货币（M2）余额291.2万亿元。"
    )
    with pytest.raises(ValueError):
        module().extract_pboc_release(content, PBC_URL)
    content = financial_document(
        "2023年11月金融统计数据报告", "11月末，广义货币（M2）余额291.2万亿元。", mirror=True
    )
    with pytest.raises(ValueError):
        module().extract_pboc_release(
            content.replace("中国人民银行".encode(), "未知来源".encode()), GD_URL
        )
    with pytest.raises(ValueError):
        module().extract_pboc_release(
            content, "https://www.gdjr.gov.cn.evil.example/post_19054.html"
        )


def test_money_amount_from_previous_year_cannot_be_admitted_for_current_report():
    content = financial_document(
        "2023年11月金融统计数据报告", "2022年11月末，广义货币（M2）余额264.7万亿元。"
    )
    with pytest.raises(ValueError):
        module().extract_pboc_release(content, PBC_URL)


@pytest.mark.parametrize("amount", ["1e999", "1e-999", "NaN", "Inf"])
def test_unrepresentable_cpi_cannot_silently_overflow_or_underflow(amount):
    content = document(
        "Consumer Prices for January 2023",
        "<table><tr><td></td><td>Y/Y (%)</td></tr>"
        f"<tr><td>Consumer Prices</td><td>{amount}</td></tr></table>",
    )
    with pytest.raises(ValueError):
        module().extract_nbs_release(content, URL)
