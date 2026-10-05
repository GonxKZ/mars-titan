"""Admisión del ajuste estacional vinculado a la tabla del PMI manufacturero."""

import pytest

from mars_titan.data.macro_release_documents import extract_nbs_release

URL = "https://www.stats.gov.cn/english/PressRelease/202205/t20220506_1830262.html"
ADJUSTED = "China's Manufacturing PMI (Seasonally Adjusted)"


def document(caption="", before="", after=""):
    return (
        '<h1 class="con_titles">Purchasing Managers Index for April 2022</h1>'
        '<div class="info">National Bureau of Statistics of China 2022-05-06 13:44</div>'
        f'<div class="TRS_Editor">{before}<div><table><tbody>'
        f'<tr><td colspan="2"><p>{caption}</p></td></tr>'
        '<tr><td colspan="2"><p>Unit: %</p></td></tr>'
        "<tr><td></td><td>PMI</td></tr>"
        "<tr><td>2022-March</td><td>49.5</td></tr>"
        "<tr><td>April</td><td>47.4</td></tr>"
        f"</tbody></table></div>{after}</div>"
    ).encode()


def test_manufacturing_pmi_caption_inside_table_keeps_values_and_publication_bound():
    result = extract_nbs_release(document(ADJUSTED), URL)
    assert [(row["period_start"], row["value"]) for row in result["observations"]] == [
        ("2022-03-01", 49.5),
        ("2022-04-01", 47.4),
    ]
    assert all(row["realtime_start"] == "2022-05-06" for row in result["observations"])
    assert all(
        row["seasonal_adjustment"] == "Seasonally Adjusted" for row in result["observations"]
    )


@pytest.mark.parametrize(
    "caption",
    [
        "China's Manufacturing PMI (Not Seasonally Adjusted)",
        "China's Manufacturing PMI (Unadjusted)",
        "China's Non-manufacturing PMI (Seasonally Adjusted)",
    ],
)
def test_conflicting_table_caption_cannot_borrow_adjustment_from_preceding_label(caption):
    with pytest.raises(ValueError, match="ajuste"):
        extract_nbs_release(document(caption, before=f"<p>{ADJUSTED}</p>"), URL)


@pytest.mark.parametrize(
    "before,after",
    [
        ("", ""),
        ("<p>Seasonally Adjusted</p>", ""),
        ("<p>Non-manufacturing PMI (Seasonally Adjusted)</p>", ""),
        (f"<p>{ADJUSTED}</p><table><tr><td>Otro indicador</td></tr></table>", ""),
        ("", f"<p>{ADJUSTED}</p>"),
    ],
)
def test_missing_or_unrelated_caption_does_not_certify_manufacturing_pmi(before, after):
    with pytest.raises(ValueError, match="ajuste"):
        extract_nbs_release(document(before=before, after=after), URL)


def test_explicit_manufacturing_caption_before_wrapped_table_remains_admissible():
    result = extract_nbs_release(document(before=f"<p>{ADJUSTED}</p>"), URL)
    assert result["observations"][-1]["value"] == 47.4
