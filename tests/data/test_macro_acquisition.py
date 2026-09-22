"""Adquisición reproducible de versiones históricas macro desde el formulario ALFRED."""

import hashlib
import io
import zipfile
from datetime import date, timedelta

import pytest


def _alfred_zip(
    series_id: str,
    vintages: list[str],
    rows: list[str],
    *,
    comment: bytes = b"",
    metadata: str = (
        "Units\nIndex 1982-1984=100  1900-01-01  Current\n"
        "Seasonal Adjustment\nSeasonally Adjusted  1900-01-01  Current\n"
    ),
) -> bytes:
    readme = "Series ID: " + series_id + "\n" + metadata + "\nVintage Dates Specified:\n\n"
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
    actual = list(macro_acquisition.iter_vintages(tmp_path))
    assert all(item["native_unit"] == "Index 1982-1984=100" for item in actual)
    assert all(item["seasonal_adjustment"] == "Seasonally Adjusted" for item in actual)
    assert [
        {
            key: item[key]
            for key in (
                "indicator_id",
                "period_start",
                "realtime_start",
                "realtime_end",
                "value",
                "source_hash",
                "source_timezone",
            )
        }
        for item in actual
    ] == [
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


def test_late_batch_failure_hides_partial_series_until_successful_resume(tmp_path, monkeypatch):
    from mars_titan.data import macro_acquisition as acquisition

    monkeypatch.setattr(acquisition, "_BATCH_SIZE", 1)
    page = (
        b'<select id="form_selected_vintage_dates">'
        b'<option value="2020-02-13"/><option value="2020-03-13"/></select>'
    )
    failed = True
    downloads = []

    def request(url, *, fields=None):
        if fields is None:
            return page, "text/html", url, 200
        vintage = fields["form[selected_vintage_dates][]"][0]
        downloads.append(vintage)
        if vintage == "2020-03-13" and failed:
            return b"<html>source failure</html>", "text/html", url, 500
        requested = sorted(
            set(fields["form[selected_vintage_dates][]"])
            | set(fields["form[entered_vintage_dates]"].split())
        )
        return (
            _alfred_zip("CPIAUCSL", requested, [f"2020-01-01,100,{vintage},"]),
            "application/zip",
            url,
            200,
        )

    monkeypatch.setattr(acquisition, "_request", request)
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
        observation_start="2020-01-01",
        observation_end="2020-01-01",
        realtime_start="2020-02-13",
        realtime_end="2020-03-13",
        workers=1,
    )
    report = acquisition.acquire_catalog(catalog, tmp_path, **arguments)
    assert report["failed_series"] == 1
    assert list(acquisition.iter_vintages(tmp_path)) == []
    failed = False
    report = acquisition.acquire_catalog(catalog, tmp_path, **arguments)
    assert report["series"][0]["resumed_batches"] == 1
    assert downloads == ["2020-02-13", "2020-03-13", "2020-03-13"]
    assert len(list(acquisition.iter_vintages(tmp_path))) == 2


def _store_metadata_fixture(tmp_path, metadata):
    from mars_titan.data import macro_acquisition as acquisition

    acquisition._initialize(tmp_path, "fixture")
    content = _alfred_zip(
        "GDPC1", ["2024-01-05"], ["2023-10-01,100,2024-01-05,2024-01-12"], metadata=metadata
    )
    acquisition._store_batch(
        tmp_path,
        {"id": "us_real_gdp"},
        "fixture",
        content,
        {"2024-01-05"},
        {},
        acquisition._parse_zip(content, "GDPC1", {"2024-01-05"}),
        "2024-01-01",
        "2024-01-12",
    )
    with acquisition._connect(tmp_path) as connection:
        connection.execute(
            "INSERT INTO series VALUES (?,?,?,?,?)",
            ("us_real_gdp", "GDPC1", "complete", None, "fixture"),
        )
    return acquisition


def test_metadata_intersections_keep_native_units_continuations_and_original_interval(tmp_path):
    acquisition = _store_metadata_fixture(
        tmp_path,
        "Units\nBillions of Chained  2000-01-01  2024-01-08\n2012 Dollars\n"
        "Billions of Chained 2017 Dollars     2024-01-09 Current\n"
        "Frequency\nQuarterly  2000-01-01 Current\n"
        "Seasonal Adjustment\nSeasonally Adjusted Annual Rate  2000-01-01 Current\n",
    )
    rows = list(acquisition.iter_vintages(tmp_path))
    assert [(r["realtime_start"], r["realtime_end"], r["native_unit"]) for r in rows] == [
        ("2024-01-05", "2024-01-08", "Billions of Chained 2012 Dollars"),
        ("2024-01-09", "2024-01-12", "Billions of Chained 2017 Dollars"),
    ]
    assert all(r["original_realtime_start"] == "2024-01-05" for r in rows)
    assert all(r["original_realtime_end"] == "2024-01-12" for r in rows)
    assert all(r["value"] == 100 for r in rows)


@pytest.mark.parametrize(
    ("units", "reason"),
    [
        ("", "missing_historical_metadata"),
        (
            "Index 2000=100  2000-01-01 Current\nIndex 2017=100  2024-01-05 Current\n",
            "ambiguous_historical_metadata",
        ),
    ],
)
def test_absent_or_ambiguous_metadata_never_inherits_catalog_units(tmp_path, units, reason):
    acquisition = _store_metadata_fixture(
        tmp_path,
        "Units\n" + units + "Seasonal Adjustment\nSeasonally Adjusted 2000-01-01 Current\n",
    )
    rows = list(acquisition.iter_vintages(tmp_path))
    assert len(rows) == 1
    assert rows[0]["value"] is None
    assert rows[0]["missing_reason"] == reason
    assert rows[0]["original_value"] == 100


def test_metadata_rebuild_verifies_archives_and_preserves_a_binary_backup(tmp_path):
    acquisition = _store_metadata_fixture(
        tmp_path,
        "Units\nIndex 2017=100 2000-01-01 Current\n"
        "Seasonal Adjustment\nSeasonally Adjusted 2000-01-01 Current\n",
    )
    with acquisition._connect(tmp_path) as connection:
        connection.execute("DROP TABLE archive_metadata")
    with pytest.raises(ValueError, match="rebuild_metadata"):
        list(acquisition.iter_vintages(tmp_path))
    backup = tmp_path / "before-metadata.sqlite3"
    result = acquisition.rebuild_metadata(tmp_path, backup_path=backup)
    assert backup.is_file()
    assert result["archives"] == 1
    assert next(acquisition.iter_vintages(tmp_path))["native_unit"] == "Index 2017=100"
    (tmp_path / "raw/us_real_gdp/fixture.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="huella"):
        acquisition.rebuild_metadata(tmp_path, backup_path=tmp_path / "before-retry.sqlite3")


def test_metadata_availability_does_not_fill_an_earlier_gap(tmp_path):
    from mars_titan.data.macro import calculate_macro
    from mars_titan.data.temporal import MarketClock

    acquisition = _store_metadata_fixture(
        tmp_path,
        "Units\nBillions of Chained 2012 Dollars 2024-01-09 Current\n"
        "Seasonal Adjustment\nSeasonally Adjusted Annual Rate 2000-01-01 Current\n",
    )
    catalog = [
        {
            "id": "us_real_gdp",
            "kind": "raw",
            "frequency": "Q",
            "series_id": "GDPC1",
            "vintage_policy": "ALFRED_OR_RELEASE_ARCHIVE",
            "unit": "catalog_current_unit",
        }
    ]
    clock = MarketClock("US", "2024-01-05", "2024-01-12")
    output = calculate_macro(acquisition.iter_vintages(tmp_path), catalog, clock)
    by_date = {r["prediction_at"].date().isoformat(): r for r in output}
    assert by_date["2024-01-08"]["missing_reason"] == "missing_historical_metadata"
    assert by_date["2024-01-09"]["value"] is None
    assert by_date["2024-01-10"]["value"] == 100
    assert by_date["2024-01-10"]["available_at"] == clock.decision("2024-01-10")
    assert by_date["2024-01-10"]["unit"] == "Billions of Chained 2012 Dollars"


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
    assert "no devolvió filas" in report["series"][0]["error"].lower()


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
    assert "versiones en conflicto" in report["series"][0]["error"].lower()
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
    assert "intervalo temporal no válido" in report["series"][0]["error"]
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
    assert "ruta no segura" in report["series"][0]["error"]


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
