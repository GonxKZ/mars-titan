"""Versiones mensuales de GSCPI sin adelantar valores revisados."""

import csv
import importlib
import json
from pathlib import Path

import pytest


def module():
    try:
        return importlib.import_module("mars_titan.data.macro_gscpi")
    except ModuleNotFoundError:
        pytest.fail("Falta el lector de versiones mensuales de GSCPI")


def content():
    return (
        b"Date,Apr-22,May-22,Jun-22,Jan-24\n"
        b"31-Mar-2022,2.7,2.80,3.0,999\n"
        b"30-Apr-2022,#N/A,3.29,3.45,999\n"
        b"31-May-2022,#N/A,#N/A,3.01,999\n"
    )


def catalog():
    with Path("data/catalogs/macro-indicators.csv").open() as stream:
        return [r for r in csv.DictReader(stream) if r["id"].startswith("global_supply_pressure")]


def test_parser_keeps_month_bound_and_ignores_future_vintage_values():
    records = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    assert len(records) == 5
    assert {r["realtime_start"] for r in records} == {"2022-05-31", "2022-06-30"}
    assert all(r["availability_precision"] == "month" for r in records)
    assert all(r["publication_timestamp_verified"] is False for r in records)
    assert all(r["value"] != 999 for r in records)
    altered = content().replace(b",999", b",changed_future_value")
    other = module().parse_gscpi_vintages(altered, last_vintage="2022-06")
    for before, after in zip(records, other, strict=True):
        assert {k: v for k, v in before.items() if k != "source_hash"} == {
            k: v for k, v in after.items() if k != "source_hash"
        }


def test_monthly_versions_and_changes_enter_only_after_month_end():
    from mars_titan.data.macro import calculate_macro
    from mars_titan.data.temporal import MarketClock

    rows = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    result = calculate_macro(rows, catalog(), MarketClock("US", "2022-05-18", "2022-07-05"))
    by_key = {(r["prediction_at"].date().isoformat(), r["indicator_id"]): r for r in result}
    assert by_key["2022-05-31", "global_supply_pressure"]["value"] is None
    assert by_key["2022-06-01", "global_supply_pressure"]["value"] == 3.29
    assert by_key["2022-06-01", "global_supply_pressure_change_1m"]["value"] == pytest.approx(0.49)
    assert by_key["2022-06-30", "global_supply_pressure"]["value"] == 3.29
    assert by_key["2022-07-01", "global_supply_pressure"]["value"] == 3.01
    assert by_key["2022-07-01", "global_supply_pressure_change_1m"]["value"] == pytest.approx(-0.44)


@pytest.mark.parametrize(
    "data",
    [
        b"Date,May-22,May-22\n30-Apr-2022,1,2\n",
        b"Date,May-22\n31-May-2022,1\n",
        b"Date,May-22\n30-Apr-2022,inf\n",
        b"Date,May-22\n30-Apr-2022,1\n30-Apr-2022,1\n",
        b"Date,May-22\n31-Apr-2022,1\n",
        b"Date,May-22\n01-Apr-2022,1\n",
        b"Date,May-22\n30-Apr-2022,1,2\n",
    ],
)
def test_malformed_dates_values_and_vintages_are_rejected(data):
    with pytest.raises(ValueError):
        module().parse_gscpi_vintages(data, last_vintage="2022-05")


def test_old_versions_cannot_bypass_the_month_end_policy():
    from mars_titan.data.macro import calculate_macro
    from mars_titan.data.temporal import MarketClock

    rows = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    rows[0]["realtime_start"] = "2022-05-18"
    with pytest.raises(ValueError):
        calculate_macro(rows, catalog(), MarketClock("US", "2022-05-18", "2022-07-05"))


def test_mislabeled_vintage_cannot_use_an_earlier_month_bound():
    from mars_titan.data.macro import calculate_macro
    from mars_titan.data.temporal import MarketClock

    rows = module().parse_gscpi_vintages(content(), last_vintage="2022-06")
    rows[0]["vintage_label"] = "Jun-22"
    with pytest.raises(ValueError):
        calculate_macro(rows, catalog(), MarketClock("US", "2022-05-18", "2022-07-05"))


def test_empty_trailing_csv_row_is_not_a_missing_observation():
    records = module().parse_gscpi_vintages(content() + b",,,,\n", last_vintage="2022-06")
    assert len(records) == 5


def test_archive_missing_its_first_regular_version_is_not_complete():
    with pytest.raises(ValueError, match="archivo|versión|mes"):
        module().parse_gscpi_vintages(b"Date,Jun-22\n30-Apr-2022,3.45\n", last_vintage="2022-06")


def test_preparation_writes_immutable_source_and_auditable_panel(tmp_path, monkeypatch):
    from mars_titan.data.storage import sha256

    monkeypatch.setattr(module(), "_download", lambda: content(), raising=False)
    output = tmp_path / "prepared"
    result = module().prepare_gscpi(
        output,
        Path("data/catalogs/macro-indicators.csv"),
        market="US",
        start="2022-05-18",
        end="2022-07-05",
    )
    assert result["source_sha256"] == sha256(output / "source.csv")
    assert result["sha256"] == sha256(output / "macro.parquet")
    assert result["publication_timestamp_verified"] is False
    assert result["nonempty_indicators"] == [
        "global_supply_pressure",
        "global_supply_pressure_change_1m",
    ]
    assert json.loads((output / "report.json").read_text()) == result
    before = (output / "report.json").read_bytes()
    with pytest.raises(FileExistsError):
        module().prepare_gscpi(
            output,
            Path("data/catalogs/macro-indicators.csv"),
            market="US",
            start="2022-05-18",
            end="2022-07-05",
        )
    assert (output / "report.json").read_bytes() == before


def test_failed_source_never_publishes_a_completed_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(module(), "_download", lambda: b"invalid source", raising=False)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError):
        module().prepare_gscpi(
            output,
            Path("data/catalogs/macro-indicators.csv"),
            market="US",
            start="2022-05-18",
            end="2022-07-05",
        )
    assert not output.exists()


def test_cli_prepares_the_real_pipeline_with_declared_month_precision(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(module(), "_download", lambda: content())
    assert (
        module().main(
            [
                "--catalog",
                "data/catalogs/macro-indicators.csv",
                "--output",
                str(tmp_path / "cli"),
                "--market",
                "US",
                "--start",
                "2022-05-18",
                "--end",
                "2022-07-05",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["computed_values"] > 0
    assert report["publication_timestamp_verified"] is False
