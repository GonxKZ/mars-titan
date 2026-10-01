"""Contratos de adquisición pública sin red ni incorporación a la referencia."""

import hashlib
import io
import json
import subprocess
import zipfile
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from scripts import public_source_formats, refresh_public_sources


@pytest.fixture
def formats():
    return public_source_formats


@pytest.fixture
def refresh(tmp_path, monkeypatch):
    monkeypatch.setattr(refresh_public_sources, "ROOT", tmp_path)
    return refresh_public_sources


VIX = b"DATE,OPEN,HIGH,LOW,CLOSE\n09/18/2026,15,16,14,15.5\n"
RSS = b"""<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0"><channel><item><title>Comunicado</title>
<link>https://www.federalreserve.gov/newsevents/example.htm</link>
<pubDate>Fri, 18 Sep 2026 14:00:00 GMT</pubDate></item></channel></rss>"""


def archive(entries, compression=zipfile.ZIP_STORED):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=compression) as zipped:
        for name, body in entries.items():
            zipped.writestr(name, body)
    return output.getvalue()


def source(identifier="cboe_vix", **changes):
    result = {
        "id": identifier,
        "provider": "Proveedor público",
        "url": "https://example.org/catalog",
        "download_url": "https://example.org/data.csv",
        "filename": "old.csv",
        "format": "csv",
        "validator": "vix_csv",
        "status": "downloaded_validated",
        "auth": "none",
    }
    return result | changes


def catalog_file(tmp_path, entries, **policy):
    path = tmp_path / "catalog.json"
    path.write_text(
        json.dumps({"schema_version": 1, "sources": entries, "download_policy": policy}),
        encoding="utf-8",
    )
    return path


def response(body, url="https://example.org/data.csv", content_type="text/csv"):
    return body, {
        "http_status": 200,
        "effective_url": url,
        "content_type": content_type,
        "received_bytes": len(body),
        "elapsed_seconds": 0.1,
    }


def test_csv_reports_dates_rows_and_missing_cells(formats):
    result = formats.validate(
        b"sasdate,ONE,TWO\nTransform:,5,2\n1/1/2026,1.5,\n2/1/2026,2,3\n",
        "fred_csv",
        as_of=date(2026, 9, 18),
    )
    assert result["valid"] is True
    assert result["row_count"] == 2
    assert result["missing_numeric_cells"] == 1
    assert result["date_min"] == "2026-01-01"
    assert result["date_max"] == "2026-02-01"


@pytest.mark.parametrize(
    "body",
    [
        b"<html>Verification required</html>",
        b"DATE,OPEN,HIGH,LOW,CLOSE\n09/18/2026,1,2,0\n",
        b"DATE,OPEN,HIGH,LOW,CLOSE\nnot-a-date,1,2,0,1\n",
        b"DATE,OPEN,HIGH,LOW,CLOSE\n09/18/2026,1,2,0,inf\n",
        b"DATE,OPEN,HIGH,LOW,CLOSE\n09/18/2026,1,2,0,1e999\n",
        b"message,value\n2026-09-18,42\n",
    ],
)
def test_csv_rejects_wrong_content_schema_and_nonfinite_values(formats, body):
    with pytest.raises(ValueError):
        formats.validate(body, "vix_csv")


def test_ecb_only_treats_observation_value_as_numeric(formats):
    result = formats.validate(
        b"KEY,TIME_PERIOD,OBS_VALUE,OBS_STATUS\nEXR.D.USD,2026-09-18,1.15,A\n",
        "ecb_csv",
    )
    assert result["row_count"] == 1
    assert result["numeric_cells_checked"] == 1


def test_stooq_validates_daily_ohlcv_without_claiming_publication_time(formats):
    result = formats.validate(
        b"Date,Open,High,Low,Close,Volume\n2026-09-18,100,102,99,101,2000\n",
        "stooq_csv",
    )
    assert result["row_count"] == 1
    assert result["date_semantics"] == "reference_period_not_publication_time"


def test_rss_reports_explicit_utc_publication_time(formats):
    result = formats.validate(RSS, "fed_rss")
    assert result["row_count"] == 1
    assert result["published_at_min_utc"] == "2026-09-18T14:00:00+00:00"


@pytest.mark.parametrize("kind", ["fed_rss", "treasury_xml"])
def test_xml_rejects_doctype_before_entity_expansion(formats, kind):
    body = b'<!DOCTYPE rss [<!ENTITY item "expanded">]><rss>&item;</rss>'
    with pytest.raises(ValueError, match="DOCTYPE|ENTITY"):
        formats.validate(body, kind)


def test_xml_rejects_utf16_doctype_bypass(formats):
    body = '<?xml version="1.0"?><!DOCTYPE rss><rss/>'.encode("utf-16")
    with pytest.raises(ValueError):
        formats.validate(body, "fed_rss")


def test_french_archive_requires_header_and_finite_daily_factors(formats):
    good = b"Generated file\n,Mkt-RF,SMB,HML,RF\n20260918,0.2,-0.1,0.3,0.01\n"
    result = formats.validate(archive({"daily.csv": good}), "french_zip")
    assert result["row_count"] == 1
    assert result["unit"] == "percent_daily_return"
    with pytest.raises(ValueError):
        formats.validate(archive({"daily.csv": good.replace(b"0.2", b"-Infinity")}), "french_zip")


def test_zip_rejects_excessive_uncompressed_size_before_reading(formats, monkeypatch):
    monkeypatch.setattr(formats, "MAX_UNCOMPRESSED_BYTES", 128)
    packed = archive({"data.csv": b"0" * 1024}, zipfile.ZIP_DEFLATED)
    with pytest.raises(ValueError, match="descomprimid"):
        formats.validate(packed, "french_zip")


def test_xlsx_rejects_a_zip_that_is_not_a_workbook(formats):
    with pytest.raises(ValueError, match="XLSX|workbook"):
        formats.validate(archive({"page.html": "<html>error</html>"}), "gscpi_xlsx")


def test_pdf_rejects_html_before_invoking_pdfinfo(formats):
    with pytest.raises(ValueError, match="PDF"):
        formats.validate(b"<html>404</html>", "financial_pdf")


def test_default_selection_excludes_failed_authenticated_and_fixed_documents(refresh):
    rows = [
        source(),
        source("blocked", status="failed"),
        source("keyed", auth="api_key"),
        source("report", validator="financial_pdf", format="pdf"),
    ]
    assert [r["id"] for r in refresh.select_sources({"sources": rows})] == ["cboe_vix"]
    assert refresh.select_sources({"sources": rows}, ["report"])[0]["id"] == "report"
    with pytest.raises(ValueError):
        refresh.select_sources({"sources": rows}, ["blocked"])


def test_dynamic_urls_roll_ecb_window_and_treasury_month(refresh):
    ecb = source(
        "ecb_usd_eur",
        download_url="https://data-api.ecb.europa.eu/data?startPeriod=2026-08-01"
        "&endPeriod=2026-09-18&format=csvdata",
    )
    query = parse_qs(urlsplit(refresh.resolve_url(ecb, date(2027, 1, 15))).query)
    assert query == {
        "startPeriod": ["2026-10-18"],
        "endPeriod": ["2027-01-15"],
        "format": ["csvdata"],
    }
    treasury = source(
        "treasury_yields",
        download_url="https://home.treasury.gov/xml?data=daily_treasury_yield_curve"
        "&field_tdr_date_value_month=202609",
    )
    query = parse_qs(urlsplit(refresh.resolve_url(treasury, date(2027, 1, 15))).query)
    assert query["field_tdr_date_value_month"] == ["202701"]
    assert query["data"] == ["daily_treasury_yield_curve"]


def test_fred_resolves_labelled_current_links_not_old_catalog_urls(refresh):
    page = b"""<a href="/research/monthly/2027-01-md.csv"><b>current.csv</b></a>
    <a href="/research/monthly/2026-08-md.csv">2026-08.csv</a>
    <a href="/research/quarterly/2027-01-qd.csv">current.csv</a>"""
    item = source("fred_md", url="https://www.stlouisfed.org/research/fred-databases")
    assert refresh.resolve_url(item, date(2027, 1, 15), page_html=page) == (
        "https://www.stlouisfed.org/research/monthly/2027-01-md.csv"
    )
    item["id"] = "fred_qd"
    assert refresh.resolve_url(item, date(2027, 1, 15), page_html=page).endswith(
        "/quarterly/2027-01-qd.csv"
    )
    with pytest.raises(ValueError, match="current"):
        refresh.resolve_url(item, date(2027, 1, 15), page_html=b"<html>No current link</html>")


@pytest.mark.parametrize("url", ["http://example.org/a.csv", "file:///etc/passwd"])
def test_catalog_url_cannot_disable_https(refresh, url):
    with pytest.raises(ValueError, match="HTTPS"):
        refresh.resolve_url(source(download_url=url), date(2026, 9, 18))


def test_list_does_not_request_network_or_create_output(refresh, tmp_path, monkeypatch, capsys):
    catalog = catalog_file(tmp_path, [source()])
    output = tmp_path / "snapshots"
    monkeypatch.setattr(refresh, "CATALOG", catalog)
    monkeypatch.setattr(refresh, "OUTPUT_ROOT", output)

    def no_network(*args, **kwargs):
        pytest.fail("--list no debe acceder a la red")

    monkeypatch.setattr(refresh, "fetch_url", no_network)
    assert refresh.main(["--list"]) == 0
    assert "cboe_vix" in capsys.readouterr().out
    assert not output.exists()


def test_snapshot_is_immutable_and_records_hash_and_non_benchmark_status(
    refresh, tmp_path, monkeypatch
):
    catalog = catalog_file(tmp_path, [source()])
    monkeypatch.setattr(refresh, "fetch_url", lambda *a, **k: response(VIX))
    when = datetime(2026, 9, 18, 15, 0, 0, 123456, tzinfo=UTC)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots", run_at=when)
    manifest_path = result["manifest_path"]
    saved = json.loads(manifest_path.read_text())
    record = saved["records"][0]
    assert record["status"] == "downloaded_validated"
    assert record["benchmark_eligible"] is False
    assert record["sha256"] == hashlib.sha256(VIX).hexdigest()
    assert record["bytes"] == len(VIX)
    assert (tmp_path / record["local_path"]).read_bytes() == VIX
    before = manifest_path.read_bytes()
    with pytest.raises(FileExistsError):
        refresh.run_refresh(catalog, tmp_path / "snapshots", run_at=when)
    assert manifest_path.read_bytes() == before


def test_bad_payload_leaves_no_valid_data_file_but_other_source_can_succeed(
    refresh, tmp_path, monkeypatch
):
    catalog = catalog_file(tmp_path, [source("bad"), source("good")])
    payloads = iter([b"<html>Rate limited</html>", VIX])
    monkeypatch.setattr(refresh, "fetch_url", lambda *a, **k: response(next(payloads)))
    monkeypatch.setattr(refresh.time, "sleep", lambda _: None)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots")
    manifest = json.loads(result["manifest_path"].read_text())
    assert [r["status"] for r in manifest["records"]] == ["failed", "downloaded_validated"]
    assert not (result["manifest_path"].parent / "bad.csv").exists()
    assert (result["manifest_path"].parent / "good.csv").read_bytes() == VIX


def test_http_failure_keeps_receipt_without_publishing_payload(refresh, tmp_path, monkeypatch):
    catalog = catalog_file(tmp_path, [source()])

    def denied(*args, **kwargs):
        raise refresh.FetchError("HTTP 403", {"http_status": 403, "received_bytes": 12})

    monkeypatch.setattr(refresh, "fetch_url", denied)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots")
    manifest = json.loads(result["manifest_path"].read_text())
    assert manifest["records"][0]["http_status"] == 403
    assert manifest["records"][0]["local_path"] is None
    assert list(result["manifest_path"].parent.iterdir()) == [result["manifest_path"]]


def test_run_budget_counts_payloads_and_prevents_excess_download(refresh, tmp_path, monkeypatch):
    catalog = catalog_file(tmp_path, [source("one"), source("two")], max_total_bytes=len(VIX))
    monkeypatch.setattr(refresh, "fetch_url", lambda *a, **k: response(VIX))
    monkeypatch.setattr(refresh.time, "sleep", lambda _: None)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots")
    manifest = json.loads(result["manifest_path"].read_text())
    assert manifest["transferred_bytes"] == len(VIX)
    assert [r["status"] for r in manifest["records"]] == ["downloaded_validated", "failed"]
    assert not (result["manifest_path"].parent / "two.csv").exists()


def test_fred_refresh_captures_resolved_sources_and_reuses_discovery_page(
    refresh, tmp_path, monkeypatch
):
    page_url = "https://www.stlouisfed.org/research/fred-databases"
    page = b"""<a href="/monthly/2027-01-md.csv">current.csv</a>
    <a href="/quarterly/2027-01-qd.csv">current.csv</a>"""
    data = b"sasdate,VALUE\n1/1/2027,42\n"
    catalog = catalog_file(
        tmp_path,
        [
            source("fred_md", url=page_url, validator="fred_csv"),
            source("fred_qd", url=page_url, validator="fred_csv"),
        ],
    )
    payloads = {
        page_url: response(page, page_url, "text/html"),
        "https://www.stlouisfed.org/monthly/2027-01-md.csv": response(
            data, "https://www.stlouisfed.org/monthly/2027-01-md.csv"
        ),
        "https://www.stlouisfed.org/quarterly/2027-01-qd.csv": response(
            data, "https://www.stlouisfed.org/quarterly/2027-01-qd.csv"
        ),
    }
    monkeypatch.setattr(refresh, "fetch_url", lambda url, *a: payloads.pop(url))
    monkeypatch.setattr(refresh.time, "sleep", lambda _: None)
    result = refresh.run_refresh(
        catalog, tmp_path / "snapshots", run_at=datetime(2027, 1, 15, tzinfo=UTC)
    )
    manifest = json.loads(result["manifest_path"].read_text())
    assert [r["status"] for r in manifest["records"]] == ["downloaded_validated"] * 2
    assert manifest["records"][0]["requested_url"].endswith("/monthly/2027-01-md.csv")
    assert manifest["records"][1]["requested_url"].endswith("/quarterly/2027-01-qd.csv")
    assert manifest["transferred_bytes"] == len(page) + 2 * len(data)


@pytest.mark.parametrize(
    "bad_response",
    [response(VIX, "http://example.org/data.csv"), response(VIX * 2)],
)
def test_insecure_redirect_and_over_budget_body_cannot_create_healthy_artifact(
    refresh, tmp_path, monkeypatch, bad_response
):
    catalog = catalog_file(tmp_path, [source()], max_source_bytes=len(VIX))
    monkeypatch.setattr(refresh, "fetch_url", lambda *a, **k: bad_response)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots")
    record = result["manifest"]["records"][0]
    assert record["status"] == "failed"
    assert record["local_path"] is None
    assert not (result["manifest_path"].parent / "cboe_vix.csv").exists()


def test_exclusive_publish_preserves_existing_file(refresh, tmp_path):
    target = tmp_path / "data.csv"
    target.write_bytes(b"prior snapshot")
    with pytest.raises(FileExistsError):
        refresh.write_new_file(target, VIX)
    assert target.read_bytes() == b"prior snapshot"
    assert list(tmp_path.iterdir()) == [target]


def test_gscpi_retains_vintage_labels_as_month_precision(formats):
    result = formats.validate(b"Date,Jan-26,Feb-26\n31-Jan-1997,0.1,0.2\n", "gscpi_csv")
    assert result["vintage_labels"] == ["Jan-26", "Feb-26"]
    assert result["vintage_timestamp_precision"] == "month_label_only"


def test_treasury_validates_numeric_yields_and_rejects_infinity(formats):
    xml = b"""<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:m="http://schemas.microsoft.com/ado/2007/08/dataservices/metadata"
      xmlns:d="http://schemas.microsoft.com/ado/2007/08/dataservices">
      <entry><content><m:properties><d:NEW_DATE>2026-09-18T00:00:00</d:NEW_DATE>
      <d:BC_10YEAR>4.1</d:BC_10YEAR></m:properties></content></entry></feed>"""
    assert formats.validate(xml, "treasury_xml")["row_count"] == 1
    with pytest.raises(ValueError):
        formats.validate(xml.replace(b">4.1<", b">inf<"), "treasury_xml")


def test_xlsx_container_rejects_doctype_even_in_auxiliary_xml(formats):
    workbook = """<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheets><sheet name="Data" sheetId="1"/></sheets></workbook>"""
    sheet = """<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
    <sheetData><row><c><v>1.5</v></c></row></sheetData></worksheet>"""
    members = {
        "[Content_Types].xml": "<Types/>",
        "xl/workbook.xml": workbook,
        "xl/worksheets/sheet1.xml": sheet,
    }
    assert formats.validate(archive(members), "gscpi_xlsx")["sheet_names"] == ["Data"]
    members["xl/styles.xml"] = '<!DOCTYPE styles [<!ENTITY a "b">]><styles/>'
    with pytest.raises(ValueError, match="DOCTYPE|ENTITY"):
        formats.validate(archive(members), "gscpi_xlsx")


def test_manifest_paths_do_not_expose_absolute_workspace_paths(refresh, tmp_path, monkeypatch):
    monkeypatch.setattr(refresh, "ROOT", tmp_path)
    catalog = catalog_file(tmp_path, [source()])
    monkeypatch.setattr(refresh, "fetch_url", lambda *a, **k: response(VIX))
    when = datetime(2026, 9, 18, 16, 0, 0, tzinfo=UTC)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots", run_at=when)
    manifest = result["manifest"]
    assert manifest["source_catalog"] == "catalog.json"
    assert manifest["records"][0]["local_path"] == (
        "snapshots/20260918T160000.000000Z/cboe_vix.csv"
    )


def test_transport_disables_personal_curl_config_and_restricts_redirects(refresh, monkeypatch):
    commands = []

    def curl_fixture(command, **kwargs):
        commands.append(command)
        Path(command[command.index("--output") + 1]).write_bytes(VIX)
        return subprocess.CompletedProcess(
            command,
            0,
            json.dumps(
                {
                    "http_code": 200,
                    "url_effective": "https://example.org/data.csv",
                    "content_type": "text/csv",
                    "size_download": len(VIX),
                    "time_total": 0.01,
                }
            ),
            "",
        )

    monkeypatch.setattr(refresh.subprocess, "run", curl_fixture)
    body, receipt = refresh.fetch_url("https://example.org/data.csv", 1024, 45, "public-research")
    assert body == VIX and receipt["http_status"] == 200
    assert commands[0][:2] == ["curl", "--disable"]
    assert commands[0][commands[0].index("--proto-redir") + 1] == "=https"
    assert "--retry" not in commands[0]


@pytest.mark.parametrize("http_status", [403, 429])
def test_denied_host_is_not_requested_again_but_other_provider_continues(
    refresh, tmp_path, monkeypatch, http_status
):
    denied_url = "https://example.org/one.csv"
    same_host_url = "https://example.org/two.csv"
    other_url = "https://other.example/data.csv"
    catalog = catalog_file(
        tmp_path,
        [
            source("denied", download_url=denied_url),
            source("same_host", download_url=same_host_url),
            source("other", download_url=other_url),
        ],
    )
    requested = []

    def provider(url, *args):
        requested.append(url)
        if url == denied_url:
            raise refresh.FetchError(
                "Proveedor no disponible",
                {
                    "http_status": http_status,
                    "effective_url": denied_url,
                    "received_bytes": 12,
                },
            )
        return response(VIX, url)

    monkeypatch.setattr(refresh, "fetch_url", provider)
    monkeypatch.setattr(refresh.time, "sleep", lambda _: None)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots")
    records = result["manifest"]["records"]
    assert [r["status"] for r in records] == ["failed", "failed", "downloaded_validated"]
    assert requested == [denied_url, other_url]
    assert records[1]["http_status"] is None
    assert records[1]["prior_http_status"] == http_status
    assert not (result["manifest_path"].parent / "same_host.csv").exists()
    assert (result["manifest_path"].parent / "other.csv").read_bytes() == VIX


def test_corrupt_deflate_is_recorded_and_does_not_abort_the_next_source(
    refresh, tmp_path, monkeypatch
):
    packed = bytearray(
        archive(
            {"daily.csv": b",Mkt-RF,SMB,HML,RF\n20260918,0.2,0.1,0.3,0.01\n"}, zipfile.ZIP_DEFLATED
        )
    )
    filename_size = int.from_bytes(packed[26:28], "little")
    extra_size = int.from_bytes(packed[28:30], "little")
    packed[30 + filename_size + extra_size] = 7  # Tipo de bloque DEFLATE reservado.
    catalog = catalog_file(
        tmp_path,
        [
            source("bad_zip", validator="french_zip", format="zip_csv"),
            source("good"),
        ],
    )
    payloads = iter([bytes(packed), VIX])
    monkeypatch.setattr(refresh, "fetch_url", lambda *a, **k: response(next(payloads)))
    monkeypatch.setattr(refresh.time, "sleep", lambda _: None)
    result = refresh.run_refresh(catalog, tmp_path / "snapshots")
    saved = json.loads(result["manifest_path"].read_text())
    assert [r["status"] for r in saved["records"]] == ["failed", "downloaded_validated"]
    assert not (result["manifest_path"].parent / "bad_zip.zip").exists()
    assert (result["manifest_path"].parent / "good.csv").read_bytes() == VIX


def test_zip_rejects_unneeded_compression_backends(formats):
    packed = archive(
        {"daily.csv": b",Mkt-RF,SMB,HML,RF\n20260918,0.2,0.1,0.3,0.01\n"}, zipfile.ZIP_BZIP2
    )
    with pytest.raises(ValueError, match="compresi"):
        formats.validate(packed, "french_zip")


def test_subsecond_capture_does_not_finish_before_it_started(refresh, tmp_path, monkeypatch):
    started = datetime(2026, 9, 18, 19, 33, 26, 801397, tzinfo=UTC)
    observed = datetime(2026, 9, 18, 19, 33, 26, 900000, tzinfo=UTC)

    class FixedClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return observed.astimezone(tz)

    monkeypatch.setattr(refresh, "datetime", FixedClock)
    monkeypatch.setattr(refresh, "fetch_url", lambda *a, **k: response(VIX))
    catalog = catalog_file(tmp_path, [source()])
    result = refresh.run_refresh(catalog, tmp_path / "snapshots", run_at=started)
    manifest = json.loads(result["manifest_path"].read_text())
    assert datetime.fromisoformat(manifest["finished_at_utc"]) >= datetime.fromisoformat(
        manifest["started_at_utc"]
    )
    assert datetime.fromisoformat(manifest["finished_at_utc"]) == observed
