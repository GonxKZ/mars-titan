"""Adquisición reproducible de vintages macro desde el formulario ALFRED."""

import hashlib
import io
import zipfile
from datetime import date, timedelta


def _alfred_zip(
    series_id: str, vintages: list[str], rows: list[str], *, comment: bytes = b""
) -> bytes:
    readme = "Series ID: " + series_id + "\n\nVintage Dates Specified:\n\n"
    readme += "\n".join(f"----------\n{vintage}\n----------" for vintage in vintages)
    header = f"period_start_date,{series_id},realtime_start_date,realtime_end_date\n"
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.comment = comment
        archive.writestr("README.txt", readme)
        archive.writestr("obs._by_real-time_period.csv", header + "\n".join(rows) + "\n")
    return stream.getvalue()


def test_acquire_catalog_preserves_real_intervals_carry_in_and_na(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b"""
    <select id="form_selected_vintage_dates" name="form[selected_vintage_dates][]">
      <option value="2019-02-11">2019-02-11</option>
      <option value="2020-02-11">2020-02-11</option>
    </select>
    """

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        requested = fields["form[selected_vintage_dates][]"]
        entered = fields["form[entered_vintage_dates]"].split()
        body = _alfred_zip(
            "CPIAUCSL",
            sorted(set(requested + entered)),
            [
                "2018-01-01,249.245,2018-02-14,2019-02-10",
                "2018-01-01,248.884,2019-02-11,2020-02-10",
                "2018-01-01,.,2020-02-11,",
            ],
        )
        return body, "application/zip", url, 200

    monkeypatch.setattr(macro_acquisition, "_request", request)
    catalog = [
        {
            "id": "us_cpi",
            "kind": "raw",
            "provider": "BLS via FRED",
            "series_id": "CPIAUCSL",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
            "verification_status": "verified_metadata_not_ingested",
        }
    ]

    report = macro_acquisition.acquire_catalog(
        catalog,
        tmp_path,
        observation_start=date(2018, 1, 1),
        observation_end=date(2018, 1, 1),
        realtime_start=date(2019, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["complete"] is True
    assert report["completed_series"] == 1
    assert list(macro_acquisition.iter_vintages(tmp_path)) == [
        {
            "indicator_id": "us_cpi",
            "period_start": "2018-01-01",
            "realtime_start": "2018-02-14",
            "realtime_end": "2019-02-10",
            "value": 249.245,
            "source_hash": report["series"][0]["source_hashes"][0],
            "source_timezone": "America/New_York",
        },
        {
            "indicator_id": "us_cpi",
            "period_start": "2018-01-01",
            "realtime_start": "2019-02-11",
            "realtime_end": "2020-02-10",
            "value": 248.884,
            "source_hash": report["series"][0]["source_hashes"][0],
            "source_timezone": "America/New_York",
        },
        {
            "indicator_id": "us_cpi",
            "period_start": "2018-01-01",
            "realtime_start": "2020-02-11",
            "realtime_end": "9999-12-31",
            "value": None,
            "source_hash": report["series"][0]["source_hashes"][0],
            "source_timezone": "America/New_York",
        },
    ]


def test_corrupt_completed_batch_is_downloaded_again_with_current_hash(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b"""
    <select id="form_selected_vintage_dates">
      <option value="2020-02-13">2020-02-13</option>
    </select>
    """
    downloads = 0

    def request(url, *, fields=None):
        nonlocal downloads
        if fields is None:
            return page, "text/html", url, 200
        downloads += 1
        body = _alfred_zip(
            "CPIAUCSL",
            ["2020-02-13", "2020-12-31"],
            ["2020-01-01,258.820,2020-02-13,2021-02-07"],
            comment=b"first" if downloads == 1 else b"recovered",
        )
        return body, "application/zip", url, 200

    monkeypatch.setattr(macro_acquisition, "_request", request)
    catalog = [
        {
            "id": "us_cpi",
            "kind": "raw",
            "provider": "BLS via FRED",
            "series_id": "CPIAUCSL",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
            "verification_status": "verified_metadata_not_ingested",
        }
    ]
    arguments = dict(
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 1, 1),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )
    macro_acquisition.acquire_catalog(catalog, tmp_path, **arguments)
    raw = next((tmp_path / "raw").rglob("*.zip"))
    raw.write_bytes(b"corrupt")

    report = macro_acquisition.acquire_catalog(catalog, tmp_path, **arguments)

    assert report["complete"] is True
    assert downloads == 2
    expected_hash = hashlib.sha256(raw.read_bytes()).hexdigest()
    assert next(macro_acquisition.iter_vintages(tmp_path))["source_hash"] == expected_hash


def test_realtime_start_before_alfred_coverage_is_clamped_and_reported(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    first_vintage = "1996-12-12"
    page = f"""
    <select id="form_selected_vintage_dates">
      <option value="{first_vintage}">{first_vintage}</option>
      <option value="1997-01-15">1997-01-15</option>
    </select>
    """.encode()

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        entered = fields["form[entered_vintage_dates]"].split()
        if "1990-01-01" in entered:
            return b"", "text/html", url, 500
        requested = sorted(set(fields["form[selected_vintage_dates][]"]) | set(entered))
        return (
            _alfred_zip(
                "CPILFESL",
                requested,
                ["1996-11-01,158.700,1996-12-12,1997-01-14"],
            ),
            "application/zip",
            url,
            200,
        )

    monkeypatch.setattr(macro_acquisition, "_request", request)
    report = macro_acquisition.acquire_catalog(
        [
            {
                "id": "us_core_cpi",
                "kind": "raw",
                "provider": "BLS via FRED",
                "series_id": "CPILFESL",
                "source_url": "https://fred.stlouisfed.org/series/CPILFESL",
                "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
                "verification_status": "verified_metadata_not_ingested",
            }
        ],
        tmp_path,
        observation_start=date(1988, 1, 1),
        observation_end=date(2025, 3, 31),
        realtime_start=date(1990, 1, 1),
        realtime_end=date(2025, 3, 31),
        workers=1,
    )

    series = report["series"][0]
    assert report["complete"] is True
    assert series["effective_realtime_start"] == first_vintage
    assert series["coverage_limited_at_start"] is True


def test_observation_range_is_clamped_to_values_declared_by_form(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b"""
    <input id="form_obs_start_date" value="2009-11-01">
    <input id="form_obs_end_date" value="2025-03-01">
    <select id="form_selected_vintage_dates">
      <option value="2014-02-19">2014-02-19</option>
    </select>
    """

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        if fields["form[obs_start_date]"] < "2009-11-01":
            return (
                b"<html>Observation start date can not be before 2009-11-01</html>",
                "text/html",
                url,
                200,
            )
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        return (
            _alfred_zip(
                "PPIFIS",
                requested,
                ["2009-11-01,100.000,2014-02-19,"],
            ),
            "application/zip",
            url,
            200,
        )

    monkeypatch.setattr(macro_acquisition, "_request", request)
    report = macro_acquisition.acquire_catalog(
        [
            {
                "id": "us_ppi_final",
                "kind": "raw",
                "provider": "BLS via FRED",
                "series_id": "PPIFIS",
                "source_url": "https://fred.stlouisfed.org/series/PPIFIS",
                "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
                "verification_status": "verified_metadata_not_ingested",
            }
        ],
        tmp_path,
        observation_start=date(1988, 1, 1),
        observation_end=date(2025, 3, 31),
        realtime_start=date(1990, 1, 1),
        realtime_end=date(2025, 3, 31),
        workers=1,
    )

    series = report["series"][0]
    assert report["complete"] is True
    assert series["effective_observation_start"] == "2009-11-01"
    assert series["effective_observation_end"] == "2025-03-01"
    assert series["observation_coverage_limited"] is True


def test_all_discovered_vintages_are_batched_with_interval_boundaries(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    first = date(2020, 1, 1)
    vintages = [(first + timedelta(days=index)).isoformat() for index in range(400)]
    page = (
        '<select id="form_selected_vintage_dates">'
        + "".join(f'<option value="{value}">{value}</option>' for value in vintages)
        + "</select>"
    ).encode()
    payloads = []

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        payloads.append(fields)
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        return (
            _alfred_zip(
                "CPIAUCSL",
                requested,
                [f"2020-01-01,258.820,{vintages[0]},9999-12-31"],
            ),
            "application/zip",
            url,
            200,
        )

    monkeypatch.setattr(macro_acquisition, "_request", request)
    catalog = [
        {
            "id": "us_cpi",
            "kind": "raw",
            "provider": "BLS via FRED",
            "series_id": "CPIAUCSL",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
            "verification_status": "verified_metadata_not_ingested",
        }
    ]

    report = macro_acquisition.acquire_catalog(
        catalog,
        tmp_path,
        observation_start=first,
        observation_end=first,
        realtime_start=first,
        realtime_end=first + timedelta(days=399),
        workers=1,
    )

    series = report["series"][0]
    assert series["vintage_date_first"] == vintages[0]
    assert series["vintage_date_last"] == vintages[-1]
    assert [len(payload["form[selected_vintage_dates][]"]) for payload in payloads] == [350, 50]
    assert payloads[0]["form[entered_vintage_dates]"].split() == [vintages[0], vintages[-1]]
    assert payloads[1]["form[entered_vintage_dates]"] == ""


def test_series_without_any_admitted_rows_is_not_complete(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b"""
    <select id="form_selected_vintage_dates">
      <option value="2020-02-13">2020-02-13</option>
    </select>
    """

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        return _alfred_zip("CPIAUCSL", requested, []), "application/zip", url, 200

    monkeypatch.setattr(macro_acquisition, "_request", request)
    catalog = [
        {
            "id": "us_cpi",
            "kind": "raw",
            "provider": "BLS via FRED",
            "series_id": "CPIAUCSL",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
            "verification_status": "verified_metadata_not_ingested",
        }
    ]

    report = macro_acquisition.acquire_catalog(
        catalog,
        tmp_path,
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 1, 1),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["complete"] is False
    assert report["failed_series"] == 1
    assert report["series"][0]["status"] == "error"
    assert "no rows" in report["series"][0]["error"].lower()


def test_identical_duplicate_vintage_is_counted_once(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b"""
    <select id="form_selected_vintage_dates">
      <option value="2020-02-13">2020-02-13</option>
    </select>
    """
    row = "2020-01-01,258.820,2020-02-13,2021-02-07"

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        return _alfred_zip("CPIAUCSL", requested, [row, row]), "application/zip", url, 200

    monkeypatch.setattr(macro_acquisition, "_request", request)
    catalog = [
        {
            "id": "us_cpi",
            "kind": "raw",
            "provider": "BLS via FRED",
            "series_id": "CPIAUCSL",
            "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
            "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
            "verification_status": "verified_metadata_not_ingested",
        }
    ]

    report = macro_acquisition.acquire_catalog(
        catalog,
        tmp_path,
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 1, 1),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["series"][0]["rows"] == 1
    assert len(list(macro_acquisition.iter_vintages(tmp_path))) == 1


def test_conflicting_duplicate_vintage_marks_series_as_error(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b'<select id="form_selected_vintage_dates"><option value="2020-02-13"/></select>'

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        return (
            _alfred_zip(
                "CPIAUCSL",
                requested,
                [
                    "2020-01-01,258.820,2020-02-13,2021-02-07",
                    "2020-01-01,999.000,2020-02-13,2021-02-07",
                ],
            ),
            "application/zip",
            url,
            200,
        )

    monkeypatch.setattr(macro_acquisition, "_request", request)
    report = macro_acquisition.acquire_catalog(
        [
            {
                "id": "us_cpi",
                "kind": "raw",
                "provider": "BLS via FRED",
                "series_id": "CPIAUCSL",
                "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
                "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
                "verification_status": "verified_metadata_not_ingested",
            }
        ],
        tmp_path,
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 1, 1),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["complete"] is False
    assert "conflicting vintages" in report["series"][0]["error"].lower()
    assert list(macro_acquisition.iter_vintages(tmp_path)) == []


def test_observation_after_realtime_start_is_rejected(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b'<select id="form_selected_vintage_dates"><option value="2020-02-13"/></select>'

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        return (
            _alfred_zip(
                "DCOILWTICO",
                requested,
                ["2020-02-14,50.000,2020-02-13,"],
            ),
            "application/zip",
            url,
            200,
        )

    monkeypatch.setattr(macro_acquisition, "_request", request)
    report = macro_acquisition.acquire_catalog(
        [
            {
                "id": "wti_spot",
                "kind": "raw",
                "provider": "EIA via FRED",
                "series_id": "DCOILWTICO",
                "source_url": "https://fred.stlouisfed.org/series/DCOILWTICO",
                "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
                "verification_status": "verified_metadata_not_ingested",
            }
        ],
        tmp_path,
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 12, 31),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["complete"] is False
    assert "invalid temporal interval" in report["series"][0]["error"]
    assert list(macro_acquisition.iter_vintages(tmp_path)) == []


def test_html_download_error_is_not_reported_as_success(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b'<select id="form_selected_vintage_dates"><option value="2020-02-13"/></select>'

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        return b'<html><p class="error">invalid vintage</p></html>', "text/html", url, 200

    monkeypatch.setattr(macro_acquisition, "_request", request)
    report = macro_acquisition.acquire_catalog(
        [
            {
                "id": "us_cpi",
                "kind": "raw",
                "provider": "BLS via FRED",
                "series_id": "CPIAUCSL",
                "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
                "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
                "verification_status": "verified_metadata_not_ingested",
            }
        ],
        tmp_path,
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 1, 1),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["complete"] is False
    assert report["failed_series"] == 1
    assert "html_error" in report["series"][0]["error"]


def test_zip_member_with_parent_path_is_rejected(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition

    page = b'<select id="form_selected_vintage_dates"><option value="2020-02-13"/></select>'

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        stream = io.BytesIO()
        with zipfile.ZipFile(stream, "w") as archive:
            archive.writestr("README.txt", "Vintage Dates Specified:\n" + "\n".join(requested))
            archive.writestr("../payload.csv", "unsafe")
        return stream.getvalue(), "application/zip", url, 200

    monkeypatch.setattr(macro_acquisition, "_request", request)
    report = macro_acquisition.acquire_catalog(
        [
            {
                "id": "us_cpi",
                "kind": "raw",
                "provider": "BLS via FRED",
                "series_id": "CPIAUCSL",
                "source_url": "https://fred.stlouisfed.org/series/CPIAUCSL",
                "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
                "verification_status": "verified_metadata_not_ingested",
            }
        ],
        tmp_path,
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 1, 1),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["complete"] is False
    assert "unsafe member path" in report["series"][0]["error"]


def test_non_downloadable_catalog_entries_keep_specific_exclusion_reasons(tmp_path):
    from mars_titan.data import macro_acquisition

    catalog = [
        {"id": "derived", "kind": "derived"},
        {
            "id": "model",
            "kind": "raw",
            "series_id": "NFCI",
            "vintage_policy": "MODEL_VINTAGES_ONLY",
        },
        {
            "id": "global_supply_pressure",
            "kind": "raw",
            "series_id": "GSCPI",
            "vintage_policy": "MODEL_VINTAGES_ONLY",
        },
        {
            "id": "pending",
            "kind": "raw",
            "series_id": "no identifier verified",
            "vintage_policy": "NO_VINTAGES_EXCLUDE",
        },
    ]

    report = macro_acquisition.acquire_catalog(
        catalog,
        tmp_path,
        observation_start=date(2020, 1, 1),
        observation_end=date(2020, 1, 1),
        realtime_start=date(2020, 1, 1),
        realtime_end=date(2020, 12, 31),
        workers=1,
    )

    assert report["catalog_entries"] == 4
    assert report["eligible_series"] == 0
    assert {item["indicator_id"]: item["reason"] for item in report["exclusions"]} == {
        "derived": "derived_not_downloaded",
        "model": "model_vintages_only",
        "global_supply_pressure": "gscpi_release_timestamp_unverified",
        "pending": "no_vintages_exclude",
    }
